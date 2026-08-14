/* bench.c — Gate D throughput benchmark (independent of the Python layer). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../src/surfcore.h"

static SurfEnvConfig defcfg(int nenvs) {
    SurfEnvConfig c;
    memset(&c, 0, sizeof(c));
    c.num_envs = nenvs;
    c.max_episode_ticks = 6000;
    c.lookahead_k = 8;
    c.lookahead_dt = 0.25f;
    c.spawn_mode = 0;
    c.spawn_curriculum_frac = 0.99f;
    c.yaw_rate_max_deg = 10.0f;
    c.yaw_jitter_deg = 5.0f;
    c.kill_z = -1e38f;
    c.water_fail = 1;
    c.pitch_rate_max_deg = 10.0f;
    c.lidar_w = 16; c.lidar_h = 8;            /* bench the real (eyes) path */
    c.lidar_hfov_deg = 120.0f; c.lidar_vfov_deg = 90.0f;
    c.lidar_range = 2000.0f;
    c.phys.sv_gravity = 800; c.phys.sv_airaccelerate = 100; c.phys.sv_accelerate = 5;
    c.phys.sv_friction = 4; c.phys.edgefriction = 2; c.phys.sv_stopspeed = 75;
    c.phys.sv_maxspeed = 320; c.phys.player_maxspeed = 250; c.phys.sv_maxvelocity = 2000;
    c.phys.sv_stepsize = 18; c.phys.sv_bounce = 1; c.phys.msec = 10;
    c.phys.enable_stamina = 1; c.phys.enable_bhop_cap = 1; c.phys.enable_duck = 1;
    return c;
}

/* wall-clock timer: clock() on Linux sums CPU time across threads, which would
 * wreck the multi-thread phase — use omp_get_wtime when available */
static double now_s(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static double run_phase(SurfSim* sim, int nenvs, int obs_dim, int steps,
                        const int32_t* actions, int nact,
                        float* obs, float* rew, uint8_t* done, uint8_t* trunc, float* tobs) {
    double t0 = now_s();
    for (int i = 0; i < steps; i++)
        surf_step(sim, &actions[(size_t)(i % nact) * nenvs * 6], obs, rew, done, trunc, tobs);
    double dt = now_s() - t0;
    return (double)steps * nenvs / dt;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "maps/surf_ski_2.bsp";
    int nenvs = argc > 2 ? atoi(argv[2]) : 256;
    int steps = argc > 3 ? atoi(argv[3]) : 2000;
    char err[256];
    SurfEnvConfig cfg = defcfg(nenvs);
    SurfSim* sim = surf_create(path, &cfg, err, sizeof(err));
    if (!sim) { printf("create: %s\n", err); return 2; }
    int obs_dim = surf_obs_dim(sim);
    printf("envs %d, obs_dim %d, steps %d\n", nenvs, obs_dim, steps);

    float* obs = (float*)malloc(sizeof(float) * (size_t)nenvs * obs_dim);
    float* tobs = (float*)malloc(sizeof(float) * (size_t)nenvs * obs_dim);
    float* rew = (float*)malloc(sizeof(float) * (size_t)nenvs);
    uint8_t* done = (uint8_t*)malloc((size_t)nenvs);
    uint8_t* trunc = (uint8_t*)malloc((size_t)nenvs);
    /* pre-generated random actions, 64 distinct batches */
    int nact = 64;
    int32_t* actions = (int32_t*)malloc(sizeof(int32_t) * (size_t)nact * nenvs * 6);
    unsigned rs = 12345;
    for (size_t i = 0; i < (size_t)nact * nenvs; i++) {
        rs = rs * 1664525u + 1013904223u; actions[i*6+0] = (int32_t)(rs >> 16) % 15;
        rs = rs * 1664525u + 1013904223u; actions[i*6+1] = (int32_t)(rs >> 16) % 7;
        rs = rs * 1664525u + 1013904223u; actions[i*6+2] = (int32_t)(rs >> 16) % 3;
        rs = rs * 1664525u + 1013904223u; actions[i*6+3] = (int32_t)(rs >> 16) % 3;
        rs = rs * 1664525u + 1013904223u; actions[i*6+4] = (int32_t)(rs >> 16) % 2;
        rs = rs * 1664525u + 1013904223u; actions[i*6+5] = (int32_t)(rs >> 16) % 2;
    }

    surf_reset_all(sim, 42, obs);
    /* warmup + sanity: no NaN in obs after 100 random steps */
    for (int i = 0; i < 100; i++)
        surf_step(sim, actions, obs, rew, done, trunc, tobs);
    int nan_count = 0;
    for (size_t i = 0; i < (size_t)nenvs * obs_dim; i++)
        if (obs[i] != obs[i]) nan_count++;
    printf("NaN obs after warmup: %d\n", nan_count);

#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    double sps1 = run_phase(sim, nenvs, obs_dim, steps, actions, nact, obs, rew, done, trunc, tobs);
    printf("single-thread: %.0f steps/s\n", sps1);
#ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    double spsN = run_phase(sim, nenvs, obs_dim, steps, actions, nact, obs, rew, done, trunc, tobs);
    printf("%d threads:    %.0f steps/s\n", omp_get_num_procs(), spsN);
#endif
    surf_destroy(sim);
    return nan_count == 0 && sps1 >= 100000.0 ? 0 : 1;
}

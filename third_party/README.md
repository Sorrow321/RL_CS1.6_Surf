# third_party/ — upstream engine reference sources

This directory holds read-only reference copies of the engine sources that
`docs/` cite and that `src/pm.c` was transcribed from. They are **not stored
in this repository** (see [NOTICE.md](../NOTICE.md) for licensing/provenance).

Populate it with:

```powershell
.\tools\fetch_third_party.ps1
```

The build (`build.ps1`) does **not** need these files — they exist for
reading alongside the docs and for future parity work.

#!/usr/bin/env python3
"""ssh ProxyCommand that connects with a clamped TCP MSS (path-MTU black hole workaround):
the SYN advertises MSS 1200, so the far end never sends segments a broken path drops."""
import os, selectors, socket, sys
host, port = sys.argv[1], int(sys.argv[2])
mss = int(os.environ.get("MSS_CLAMP", "1200"))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss)
s.connect((host, port))
sel = selectors.DefaultSelector()
sel.register(sys.stdin.buffer.raw, selectors.EVENT_READ, "in")
sel.register(s, selectors.EVENT_READ, "sock")
out = sys.stdout.buffer.raw
while True:
    for key, _ in sel.select():
        if key.data == "in":
            b = os.read(sys.stdin.fileno(), 65536)
            if not b:
                s.shutdown(socket.SHUT_WR); sel.unregister(sys.stdin.buffer.raw)
                continue
            s.sendall(b)
        else:
            b = s.recv(65536)
            if not b:
                sys.exit(0)
            while b:
                n = out.write(b); b = b[n:]

import os, re, sys, time
buf = b""
while True:
    d = os.read(0, 65536)
    if not d: break
    buf += d
    parts = re.split(rb"[\r\n]+", buf); buf = parts.pop()
    t = int(time.time())
    sys.stdout.write("".join("%d %s\n" % (t, p.decode(errors="replace")) for p in parts if p)); sys.stdout.flush()
if buf: print(int(time.time()), buf.decode(errors="replace"))

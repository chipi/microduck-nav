# mjlab's plugin loader imports microduck_nav.tasks as soon as mjlab is imported; importing mjlab first, here,
# keeps that from meeting a half-initialised submodule when the first import is, say, microduck_nav.obstacles.
import mjlab  # noqa: F401

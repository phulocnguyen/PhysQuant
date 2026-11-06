# Assets

`geometry.py` and `spawners.py` are independent copies of the source project code.
Small background wrapper layers are copied; references to local InteriorAgent
scenes are relative to this asset tree.

The large `baskets`, `objects*`, `textures`, and `scenes/InteriorAgent` directories
are symlinks to the existing `Spatial-Understanding/assets` directories. RoboCasa
wrappers additionally reference the existing external scene volume. These links
save disk space; keep those locations available for local simulation. Do not edit
shared assets through these links while collection jobs are using them.

`tools/bundle.py` copies linked assets and the background source scenes used by the
selected manifests into a portable directory. Build on a disk with sufficient free
space. Robot assets from NVIDIA still need the online asset root or a local cache.

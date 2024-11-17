# ggml-metalium

Experimental backend for Tenstorrent devices using the Metalium and TTNN stack.

## Device selection

GGML although supports multi-GPU, TTNN supports multi device as a native functionaliy and provides much more flexablity. Due to unable to express the full configuration through standard GGML interface, the Metalium backend uses the `GGML_METALIUM_DEVICE` enviroment variable to control device selection and clustering. By default (without the variable set), only the 1st device is reported to GGML. The configuration syntax is as follows:

```bash
# Single device
GGML_METALIUM_DEVICE=0

# Multiple devices reported to GGML (!!IMPORTANT!! does not work in inference, lacking explicit data transfer)
GGML_METALIUM_DEVICE=0,1,2,3
GGML_METALIUM_DEVICE=0-3 # Same as above
GGML_METALIUM_DEVICE=0-2,3 # Same as above

# Clustering (multiple devices acting as one, supported shape and topology depends on physical connections and tt-topology settings)
GGML_METALIUM_DEVICE="cluster(2, 2)" # 2x2 mesh (default topology is ROW_MAJOR aka mesh)
GGML_METALIUM_DEVICE="cluster(2, 2, LINEAR)" # 2x2, with linear topology
GGML_METALIUM_DEVICE="cluster(2, 2, TORUS)" # 2x2, with torus topology
GGML_METALIUM_DEVICE="cluster(2, 1, MESH, 1, 1)" # 2x1 mesh, offset by 1 device in both X and Y direction
```

**IMPORTANT NOTES:**
* The Wormhole N300 cards contains 2 chips. By default only chip 0 is used. Cluster configuration is needed to make use of both chip.
* Clustering nor multi-device is supported on Grayskull cards (e.g. e75 and e150). Wormhole or better is needed.
## A basic stl viewer built with
[DearPyGUI](https://github.com/hoffstadt/DearPyGui) and
[trimesh](https://trimesh.org/)

It's not perfect - there are some rendering issues associated with z-index
clipping, but it does a decent job.  Written to learn some 3d shading math.

## Invocation

Run with [uv](https://docs.astral.sh/uv/), ala:

```bash
git clone git@github.com:zparmley/BasicStlViewer.git
cd BasicStlViewer
uv run -m stlviewer
```

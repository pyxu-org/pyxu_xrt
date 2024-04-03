Pyxu_XRT Documentation
######################

This package provides a Pyxu operators to compute samples of the *X-Ray Transform* in 2D and 3D using different methods.

The X-Ray Transform (XRT) of a function $f: \mathbb{R}^{D} \to \mathbb{R}$ is defined as

.. math::

   \mathcal{P}[f](\mathbf{n}, \mathbf{t})
   =
   \int_{\mathbb{R}} f(\mathbf{t} + \mathbf{n} \alpha) d\alpha,

where :math:`\mathbf{n}\in \mathbb{S}^{D-1}` and :math:`\mathbf{t} \in \mathbf{n}^{\perp}`.
:math:`\mathcal{P}[f]` hence denotes the set of *line integrals* of :math:`f`.

Two class of algorithms exist to evaluate the XRT:

* **Fourier methods** leverage the [Fourier Slice Theorem (FST)](https://en.wikipedia.org/wiki/Projection-slice_theorem)
  to efficiently evaluate the XRT *when multiple values of* :math:`\mathbf{t}` *are desired for each*
  :math:`\mathbf{n}`.
* **Ray methods** compute estimates of the XRT via quadrature rules by assuming :math:`f` is piecewise constant on short
  intervals.

The operators in ``pyxu_xrt`` allow one to efficiently evaluate samples of the XRT assuming :math:`f` is a pixelized
image/volume where:

* the lower-left element of :math:`f` is located at :math:`\mathbf{o} \in \mathbb{R}^{D}`,
* pixel dimensions are :math:`\mathbf{\Delta} \in \mathbb{R}_{+}^{D}`, i.e.

.. math::

   f(\mathbf{r}) = \sum_{\{\mathbf{q}\} \subset \mathbb{N}^{D}}
                   \alpha_{\mathbf{q}}
                   1_{[\mathbf{0}, \mathbf{\Delta}]}(\mathbf{r} - \mathbf{q} \odot \mathbf{\Delta} - \mathbf{o}),
   \quad
   \alpha_{\mathbf{q}} \in \mathbb{R}.


In the 2D case, the parametrization is best explained by the figure below:
<center>
<img align="center" src="../_static/api/xray/xray_parametrization.svg" alt="2D XRay Geometry" width=35%>
</center>


Notes
-----


Installation
------------

You can install ``pyxu_xrt`` via ``pip``:

.. code-block:: bash

   pip install pyxu_xrt


Developer Install
+++++++++++++++++

The HTML docs can be built using Sphinx:

.. code-block:: bash

   pip install pyxu_xrt[dev]
   pre-commit install

   tox run -e doc   # build docs
   tox run -e dist  # build packages for PyPI


.. todo::

   * Explain how to use it via pyxu imports and not via pyxu_xrt.
   * Explain why Docker images are required.


.. toctree::
   :maxdepth: 1
   :hidden:

   api/index
   examples/index
   references

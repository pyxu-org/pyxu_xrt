pyxu_xrt
========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT
.. image:: https://img.shields.io/pypi/v/pyxu_xrt.svg?color=green
   :target: https://pypi.org/project/pyxu_xrt
   :alt: PyPI
.. .. image:: https://img.shields.io/endpoint?url=https://pyxu-org.github.io/fair/shields/pyxu_xrt
..    :alt: Pyxu score
..    :target: https://pyxu-org.github.io/fair/score.html

Pyxu X-Ray Transform Operators
------------------------------

``pyxu_xrt`` provides Pyxu operators to compute samples of the *X-Ray Transform* in 2D and 3D.

The X-Ray Transform (XRT) of a function :math:`f: \mathbb{R}^{D} \to \mathbb{R}` is defined as

.. math::

   \mathcal{P}[f](\mathbf{n}, \mathbf{t})
   =
   \int_{\mathbb{R}} f(\mathbf{t} + \mathbf{n} \alpha) d\alpha,

where :math:`\mathbf{n}\in \mathbb{S}^{D-1}` and :math:`\mathbf{t} \in \mathbf{n}^{\perp}`.
:math:`\mathcal{P}[f]` hence denotes the set of *line integrals* of :math:`f`.

Two class of algorithms exist to evaluate the XRT:

* **Fourier methods** leverage the `Fourier Slice Theorem (FST)
  <https://en.wikipedia.org/wiki/Projection-slice_theorem>`_ to efficiently evaluate the XRT *when multiple values of*
  :math:`\mathbf{t}` *are desired for each* :math:`\mathbf{n}`.
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

.. image:: https://raw.githubusercontent.com/pyxu-org/pyxu_xrt/master/doc/_static/api/xray/xray_parametrization.png
   :width: 25%
   :alt: 2D XRay Geometry

Installation
------------

You can install ``pyxu_xrt`` via `pip`_:

.. code-block:: bash

   pip install pyxu_xrt


The host system must have `CUDA 11.x or 12.x <https://docs.nvidia.com/cuda/>`_ installed to use the GPU. Similarly,
using `drjit`_-backed operators on the CPU requires `LLVM <https://llvm.org/>`_. If problems arise, we provide `Docker
receipes <https://github.com/pyxu-org/pyxu_docker>`_ to easily create Pyxu developer environments.

License
-------

Distributed under the terms of the `MIT`_ license, ``pyxu_xrt`` is free and open source software.

Issues
------

If you encounter any problems, please `file an issue`_ along with a detailed description.

.. _Pyxu: https://github.com/pyxu-org/pyxu
.. _contributing-guide: https://pyxu-org.github.io/fair/contribute.html
.. _developer notes: https://pyxu-org.github.io/fair/dev_notes.html
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT: http://opensource.org/licenses/MIT
.. _cookiecutter-pyxu: https://github.com/pyxu-org/cookiecutter-pyxu
.. _tox: https://tox.readthedocs.io/en/latest/
.. _pip: https://pypi.org/project/pip/
.. _file an issue: https://github.com/pyxu-org/pyxu_xrt/issues
.. _drjit: https://drjit.readthedocs.io/en/latest/

.. QUKKOS documentation master file, created by on Tue Aug 01 20:23:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../assets/qukkos_full_logo.svg

\
\

QUKKOS represents a new feature to enable qukkos with transparent device selection. 
For application developers, it is not easy to identify which quantum device is the most appropriate 
to use in a heterogeneous quantum hybrid system, since this depends on the characteristics of both the application 
and the hardware. In qukkos, a backend is associated with one specific programming model/quantum hardware. 
Programmers decide which quantum backend to use at compilation time. 
This new feature implemented on the XACC backend eliminates the burden of deciding which quantum device to use, 
providing a highly productive quantum hybrid programming solution for qukkos applications. 
This new qukkos feature provides high accelerations of up to many thousands times  thanks to automatic 
and transparent quantum hybrid device selection.
QUKKOS is a single-source C++, retargetable quantum-classical compiler enabling 
low and high level quantum programming, compilation, and execution. QUKKOS represents the integration 
of the XACC quantum framework with the ubiquitous Clang/LLVM classical compiler frameworks. The QUKKOS 
compiler extends both of these infrastructures via simple plugin extensions, and enables programmers 
to express quantum kernel expressions (functors containing quantum code) alongside standard 
C++. 

Description of Architecture
---------------------------

For KOKKOS class documentation, check out this `site <https://kokkos.org/kokkos-core-wiki/index.html>`_.
For XACC class documentation, check out this `site <https://ornl-qci.github.io/xacc-api-docs/>`_.

QUKKOS Development Team
----------------------
SQKQuantumLab
QUKKOS is developed and maintained by:
SQK inc

Questions, Bug Reporting, and Issue Tracking
---------------------------------------------

Questions, bug reporting and issue tracking are provided by GitHub. Please
report all bugs by creating a `new issue <https://github.com/sqkcloud/qukkos/issues/new>`_.
You can ask questions by creating a new issue with the question tag.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   install
   basics
  
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

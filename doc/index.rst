.. eScatter CSLib documentation master file, created by
   sphinx-quickstart on Mon Feb 27 13:17:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to eScatter CSLib's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

CSLib is a set of common functions and classes for the different components of
the eScatter tool chain. This includes data types for containing tables of
cross-sections, a common :py:class:`UnitRegistry` that manages conversions between
physical units, and a system for validating, parsing and generating
configuration files for different sub-systems.

Installing
==========

In the root folder of this repository, run::

        pip install .

If you plan do develop CSLib, add the `-e` flag so that changes to your local copy
are effective immediately.

Physical Units
==============

.. automodule:: cslib.units

Validation, parsing and generation of Settings
==============================================

.. automodule:: cslib.predicates
        :members:

.. automodule:: cslib.settings
        :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

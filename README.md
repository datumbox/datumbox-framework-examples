Code Examples for Datumbox Machine Learning Framework
=====================================================

[![Datumbox](http://www.datumbox.com/img/logo.png)](http://www.datumbox.com/)

This project provides examples on how to use the [Datumbox Machine Learning Framework](https://github.com/datumbox/datumbox-framework) v0.7.1-SNAPSHOT (Build 20160321).

Copyright & License
-------------------

Copyright (c) 2013-2016 [Vasilis Vryniotis](http://blog.datumbox.com/author/bbriniotis/). 

The code is licensed under the [Apache License, Version 2.0](./LICENSE).

How to use
----------

The code uses Maven Project Structure and contains the following code examples:

- [Classification.java](./src/main/java/com/datumbox/examples/Classification.java): Contains an example on how to perform Classification.
- [Clustering.java](./src/main/java/com/datumbox/examples/Clustering.java): It is an example that runs Cluster Analysis.
- [Regression.java](./src/main/java/com/datumbox/examples/Regression.java): Shows how to run Regression Analysis.
- [DataModeling.java](./src/main/java/com/datumbox/examples/DataModeling.java): Explains how to use the convenience Modeler class.
- [TextClassification.java](./src/main/java/com/datumbox/examples/TextClassification.java): Uses the convenience TextClassifier class.

All of the above files contain a main() method. To use it just clone the project on your workspace and run any of the above files.

The project contains also two configuration files in the resources folder:

- [datumbox.config.properties](./src/main/resources/datumbox.config.properties): It contains the configuration for the framework (required).
- [logback.xml](./src/main/resources/logback.xml): It contains the configuration file for the logger (optional).

Finally in the resources folder there are several [real world datasets](./src/main/resources/datasets/) which are used for testing.

Useful Links
------------

- [Datumbox Machine Learning Framework](https://github.com/datumbox/datumbox-framework/)
- [Project Description](http://blog.datumbox.com/new-open-source-machine-learning-framework-written-in-java/)
- [Datumbox.com](http://www.datumbox.com/)
- [Machine Learning Blog](http://blog.datumbox.com/)


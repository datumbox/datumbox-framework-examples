Code Examples for Datumbox Machine Learning Framework
=====================================================

[![Datumbox](http://www.datumbox.com/img/logo.png)](http://www.datumbox.com/)

This project provides examples on how to use the [Datumbox Machine Learning Framework](https://github.com/datumbox/datumbox-framework/) v0.8.1-SNAPSHOT (Build 20170114).

Copyright & License
-------------------

Copyright (c) 2013-2017 [Vasilis Vryniotis](http://blog.datumbox.com/author/bbriniotis/). 

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

The project contains also 5 configuration files in the resources folder:

- [datumbox.configuration.properties](./src/main/resources/datumbox.configuration.properties): It defines for the default storage engine (required).
- [datumbox.concurrencyconfiguration.properties](./src/main/resources/datumbox.concurrencyconfiguration.properties): It controls the concurrency levels (required).
- [datumbox.inmemoryconfiguration.properties](./src/main/resources/datumbox.inmemoryconfiguration.properties): It contains the configurations for the InMemory storage engine (required).
- [datumbox.mapdbconfiguration.properties](./src/main/resources/datumbox.mapdbconfiguration.properties): It contains the configurations for the MapDB storage engine (optional).
- [logback.xml](./src/main/resources/logback.xml): It contains the configuration file for the logger (optional).

Finally in the resources folder there are several [real world datasets](./src/main/resources/datasets/) which are used for testing.

Useful Links
------------

- [Datumbox Machine Learning Framework](https://github.com/datumbox/datumbox-framework/)
- [Datumbox Zoo: Pre-trained models](https://github.com/datumbox/datumbox-framework-zoo/)
- [Datumbox.com](http://www.datumbox.com/)
- [Machine Learning Blog](http://blog.datumbox.com/)


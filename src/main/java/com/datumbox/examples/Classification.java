/**
 * Copyright (C) 2013-2020 Vasilis Vryniotis <bbriniotis@datumbox.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.datumbox.examples;

import com.datumbox.framework.common.Configuration;
import com.datumbox.framework.core.common.dataobjects.Dataframe;
import com.datumbox.framework.core.common.dataobjects.Record;
import com.datumbox.framework.common.dataobjects.TypeInference;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.MLBuilder;
import com.datumbox.framework.core.machinelearning.classification.SoftMaxRegression;
import com.datumbox.framework.core.machinelearning.featureselection.PCA;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.ClassificationMetrics;
import com.datumbox.framework.core.machinelearning.modelselection.splitters.ShuffleSplitter;
import com.datumbox.framework.core.machinelearning.preprocessing.MinMaxScaler;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Classification example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class Classification {
    
    /**
     * Example of how to use directly the algorithms of the framework in order to
     * perform classification. A similar approach can be used to perform clustering,
     * regression, build recommender system or perform topic modeling and dimensionality
     * reduction.
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args) { 
        /**
         * There are 5 configuration files in the resources folder:
         *
         * - datumbox.configuration.properties: It defines for the default storage engine (required)
         * - datumbox.concurrencyconfiguration.properties: It controls the concurrency levels (required)
         * - datumbox.inmemoryconfiguration.properties: It contains the configurations for the InMemory storage engine (required)
         * - datumbox.mapdbconfiguration.properties: It contains the configurations for the MapDB storage engine (optional)
         * - logback.xml: It contains the configuration file for the logger (optional)
         */
        
        //Initialization
        //--------------
        RandomGenerator.setGlobalSeed(42L); //optionally set a specific seed for all Random objects
        Configuration configuration = Configuration.getConfiguration(); //default configuration based on properties file
        //configuration.setStorageConfiguration(new InMemoryConfiguration()); //use In-Memory engine (default)
        //configuration.setStorageConfiguration(new MapDBConfiguration()); //use MapDB engine
        //configuration.getConcurrencyConfiguration().setParallelized(true); //turn on/off the parallelization
        //configuration.getConcurrencyConfiguration().setMaxNumberOfThreadsPerTask(4); //set the concurrency level
        
        
        
        
        //Reading Data
        //------------
        Dataframe data;
        try (Reader fileReader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(Paths.get(Classification.class.getClassLoader().getResource("datasets/diabetes/diabetes.tsv.gz").toURI()).toFile())), "UTF-8"))) {
            LinkedHashMap<String, TypeInference.DataType> headerDataTypes = new LinkedHashMap<>();
            headerDataTypes.put("pregnancies", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("plasma glucose", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("blood pressure", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("triceps thickness", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("serum insulin", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("bmi", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("dpf", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("age", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("test result", TypeInference.DataType.CATEGORICAL);

            data = Dataframe.Builder.parseCSVFile(fileReader, "test result", headerDataTypes, '\t', '"', "\r\n", null, null, configuration);
        }
        catch(UncheckedIOException | IOException | URISyntaxException ex) {
            throw new RuntimeException(ex);
        }

        //Spit into train and test datasets
        ShuffleSplitter.Split split = new ShuffleSplitter(0.8, 1).split(data).next();
        Dataframe trainingDataframe = split.getTrain();
        Dataframe testingDataframe = split.getTest();
        
        
        //Transform Dataframe
        //-----------------
        
        //Scale continuous variables
        MinMaxScaler.TrainingParameters nsParams = new MinMaxScaler.TrainingParameters();
        MinMaxScaler numericalScaler = MLBuilder.create(nsParams, configuration);

        numericalScaler.fit_transform(trainingDataframe);
        numericalScaler.save("Diabetes");
        


        //Feature Selection
        //-----------------
        
        //Perform dimensionality reduction using PCA

        PCA.TrainingParameters featureSelectionParameters = new PCA.TrainingParameters();
        featureSelectionParameters.setMaxDimensions(trainingDataframe.xColumnSize()-1); //remove one dimension
        featureSelectionParameters.setWhitened(false);
        featureSelectionParameters.setVariancePercentageThreshold(0.99999995);

        PCA featureSelection = MLBuilder.create(featureSelectionParameters, configuration);
        featureSelection.fit_transform(trainingDataframe);
        featureSelection.save("Diabetes");
        
        
        
        //Fit the classifier
        //------------------
        
        SoftMaxRegression.TrainingParameters param = new SoftMaxRegression.TrainingParameters();
        param.setTotalIterations(200);
        param.setLearningRate(0.1);

        SoftMaxRegression classifier = MLBuilder.create(param, configuration);
        classifier.fit(trainingDataframe);
        classifier.save("Diabetes");
        
        
        //Use the classifier
        //------------------
        
        //Apply the same numerical scaling on testingDataframe
        numericalScaler.transform(testingDataframe);
        
        //Apply the same featureSelection transformations on testingDataframe
        featureSelection.transform(testingDataframe);

        //Use the classifier to make predictions on the testingDataframe
        classifier.predict(testingDataframe);
        
        //Get validation metrics on the test set
        ClassificationMetrics vm = new ClassificationMetrics(testingDataframe);
        
        System.out.println("Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Classifier Accuracy: "+vm.getAccuracy());
        
        
        
        //Clean up
        //--------
        
        //Delete scaler, featureselector and classifier.
        numericalScaler.delete();
        featureSelection.delete();
        classifier.delete();
        
        //Close Dataframes.
        trainingDataframe.close();
        testingDataframe.close();
    }
    
}

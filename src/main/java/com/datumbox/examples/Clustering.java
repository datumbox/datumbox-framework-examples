/**
 * Copyright (C) 2013-2016 Vasilis Vryniotis <bbriniotis@datumbox.com>
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
import com.datumbox.framework.common.dataobjects.Dataframe;
import com.datumbox.framework.common.dataobjects.Record;
import com.datumbox.framework.common.dataobjects.TypeInference;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.MLBuilder;
import com.datumbox.framework.core.machinelearning.clustering.Kmeans;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.ClusteringMetrics;
import com.datumbox.framework.core.machinelearning.preprocessing.CornerConstraintsEncoder;
import com.datumbox.framework.core.machinelearning.preprocessing.MinMaxScaler;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Clustering example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class Clustering {
    
    /**
     * Example of how to use directly the algorithms of the framework in order to
     * perform clustering. A similar approach can be used to perform classification,
     * regression, build recommender system or perform topic modeling and dimensionality
     * reduction.
     * 
     * @param args the command line arguments
     */
    public static void main(String[] args) {  
        /**
         * There are two configuration files in the resources folder:
         * 
         * - datumbox.configuration.properties: It contains the configuration for the storage engines (required)
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
        Dataframe trainingDataframe;
        try (Reader fileReader = new InputStreamReader(new FileInputStream(Paths.get(Clustering.class.getClassLoader().getResource("datasets/heart-desease/heart.csv").toURI()).toFile()), "UTF-8")) {
            LinkedHashMap<String, TypeInference.DataType> headerDataTypes = new LinkedHashMap<>();
            headerDataTypes.put("Age", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Sex", TypeInference.DataType.CATEGORICAL);
            headerDataTypes.put("ChestPain", TypeInference.DataType.CATEGORICAL);
            headerDataTypes.put("RestBP", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Cholesterol", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("BloodSugar", TypeInference.DataType.BOOLEAN);
            headerDataTypes.put("ECG", TypeInference.DataType.CATEGORICAL); 
            headerDataTypes.put("MaxHeartRate", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Angina", TypeInference.DataType.BOOLEAN);
            headerDataTypes.put("OldPeak", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("STSlope", TypeInference.DataType.ORDINAL);
            headerDataTypes.put("Vessels", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Thal", TypeInference.DataType.CATEGORICAL);
            headerDataTypes.put("Class", TypeInference.DataType.CATEGORICAL);
            
            trainingDataframe = Dataframe.Builder.parseCSVFile(fileReader, "Class", headerDataTypes, ',', '"', "\r\n", null, null, configuration);
        }
        catch(UncheckedIOException | IOException | URISyntaxException ex) {
            throw new RuntimeException(ex);
        }

        //Store data and load them back
        trainingDataframe.save("HeartDeseaseDataset");
        Dataframe testingDataframe = Dataframe.Builder.load("HeartDeseaseDataset", configuration);
        
        
        //Transform Dataframe
        //-----------------
        
        //Convert Categorical variables to dummy variables (boolean) and scale continuous variables
        MinMaxScaler.TrainingParameters nsParams = new MinMaxScaler.TrainingParameters();
        MinMaxScaler numericalScaler = MLBuilder.create(nsParams, configuration);

        numericalScaler.fit_transform(trainingDataframe);
        numericalScaler.save("HeartDesease");

        CornerConstraintsEncoder.TrainingParameters ceParams = new CornerConstraintsEncoder.TrainingParameters();
        CornerConstraintsEncoder categoricalEncoder = MLBuilder.create(ceParams, configuration);

        categoricalEncoder.fit_transform(trainingDataframe);
        categoricalEncoder.save("HeartDesease");
        
        
        
        //Fit the clusterer
        //-----------------
        
        Kmeans.TrainingParameters param = new Kmeans.TrainingParameters();
        param.setK(2);
        param.setMaxIterations(200);
        param.setInitializationMethod(Kmeans.TrainingParameters.Initialization.FORGY);
        param.setDistanceMethod(Kmeans.TrainingParameters.Distance.EUCLIDIAN);
        param.setWeighted(false);
        param.setCategoricalGamaMultiplier(1.0);
        param.setSubsetFurthestFirstcValue(2.0);

        Kmeans clusterer = MLBuilder.create(param, configuration);
        clusterer.fit(trainingDataframe);
        clusterer.save("HeartDesease");
        
        
        //Use the clusterer
        //-----------------
        
        //Apply the same scaling and encoding on testingDataframe
        numericalScaler.transform(testingDataframe);
        categoricalEncoder.transform(testingDataframe);

        //Make predictions on the test set
        clusterer.predict(testingDataframe);
        
        //Get validation metrics on the test set
        ClusteringMetrics vm = new ClusteringMetrics(testingDataframe);
        
        System.out.println("Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Original Y: "+r.getY()+", Predicted Cluster Id: "+r.getYPredicted());
        }
        
        System.out.println("Clusterer Purity: "+vm.getPurity());
        
        
        
        //Clean up
        //--------
        
        //Delete scaler, encoder, clusterer.
        numericalScaler.delete();
        categoricalEncoder.delete();
        clusterer.delete();
        
        //Delete the train and close the test Dataframe.
        trainingDataframe.delete();
        testingDataframe.close();
    }
    
}

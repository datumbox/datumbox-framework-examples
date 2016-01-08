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

import com.datumbox.common.dataobjects.Dataframe;
import com.datumbox.common.dataobjects.Record;
import com.datumbox.common.dataobjects.TypeInference;
import com.datumbox.common.persistentstorage.ConfigurationFactory;
import com.datumbox.common.persistentstorage.interfaces.DatabaseConfiguration;
import com.datumbox.common.utilities.PHPfunctions;
import com.datumbox.common.utilities.RandomGenerator;
import com.datumbox.framework.machinelearning.clustering.Kmeans;
import com.datumbox.framework.machinelearning.datatransformation.DummyXMinMaxNormalizer;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.HashMap;
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
         * - datumbox.config.properties: It contains the configuration for the storage engines (required)
         * - logback.xml: It contains the configuration file for the logger (optional)
         */    
        
        //Initialization
        //--------------
        RandomGenerator.setGlobalSeed(42L); //optionally set a specific seed for all Random objects
        DatabaseConfiguration dbConf = ConfigurationFactory.INMEMORY.getConfiguration(); //in-memory maps
        //DatabaseConfiguration dbConf = ConfigurationFactory.MAPDB.getConfiguration(); //mapdb maps
        
        
        
        //Reading Data
        //------------
        Dataframe trainingDataframe;
        try (Reader fileReader = new InputStreamReader(new FileInputStream(Paths.get(Clustering.class.getClassLoader().getResource("datasets/heart-desease/heart.csv").toURI()).toFile()), "UTF-8")) {
            Map<String, TypeInference.DataType> headerDataTypes = new HashMap<>();
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
            
            trainingDataframe = Dataframe.Builder.parseCSVFile(fileReader, "Class", headerDataTypes, ',', '"', "\r\n", dbConf);
        }
        catch(UncheckedIOException | IOException | URISyntaxException ex) {
            throw new RuntimeException(ex);
        }
        Dataframe testingDataframe = trainingDataframe.copy();
        
        
        //Transform Dataframe
        //-----------------
        
        //Convert Categorical variables to dummy variables (boolean) and normalize continuous variables
        DummyXMinMaxNormalizer dataTransformer = new DummyXMinMaxNormalizer("HeartDesease", dbConf);
        dataTransformer.fit_transform(trainingDataframe, new DummyXMinMaxNormalizer.TrainingParameters());
        
        
        
        //Fit the clusterer
        //-----------------
        
        Kmeans clusterer = new Kmeans("HeartDesease", dbConf);
        
        Kmeans.TrainingParameters param = new Kmeans.TrainingParameters();
        param.setK(2);
        param.setMaxIterations(200);
        param.setInitializationMethod(Kmeans.TrainingParameters.Initialization.FORGY);
        param.setDistanceMethod(Kmeans.TrainingParameters.Distance.EUCLIDIAN);
        param.setWeighted(false);
        param.setCategoricalGamaMultiplier(1.0);
        param.setSubsetFurthestFirstcValue(2.0);
        
        clusterer.fit(trainingDataframe, param);
        
        //Denormalize trainingDataframe (optional)
        dataTransformer.denormalize(trainingDataframe);
        
        System.out.println("Cluster assignments (Record Ids):");
        for(Map.Entry<Integer, Kmeans.Cluster> entry: clusterer.getClusters().entrySet()) {
            Integer clusterId = entry.getKey();
            Kmeans.Cluster cl = entry.getValue();
            
            System.out.println("Cluster "+clusterId+": "+cl.getRecordIdSet());
        }
        
        
        
        //Use the clusterer
        //-----------------
        
        //Apply the same transformations on testingDataframe
        dataTransformer.transform(testingDataframe);
        
        //Get validation metrics on the training set
        Kmeans.ValidationMetrics vm = clusterer.validate(testingDataframe);
        clusterer.setValidationMetrics(vm); //store them in the model for future reference
        
        //Denormalize testingDataframe (optional)
        dataTransformer.denormalize(testingDataframe);
        
        System.out.println("Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Original Y: "+r.getY()+", Predicted Cluster Id: "+r.getYPredicted());
        }
        
        System.out.println("Clusterer Statistics: "+PHPfunctions.var_export(vm));
        
        
        
        //Clean up
        //--------
        
        //Delete data transformer, clusterer.
        dataTransformer.delete();
        clusterer.delete();
        
        //Delete Dataframes.
        trainingDataframe.delete();
        testingDataframe.delete();
    }
    
}

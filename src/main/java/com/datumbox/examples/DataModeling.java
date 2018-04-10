/**
 * Copyright (C) 2013-2018 Vasilis Vryniotis <bbriniotis@datumbox.com>
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

import com.datumbox.framework.applications.datamodeling.Modeler;
import com.datumbox.framework.common.Configuration;
import com.datumbox.framework.core.common.dataobjects.Dataframe;
import com.datumbox.framework.core.common.dataobjects.Record;
import com.datumbox.framework.common.dataobjects.TypeInference;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.MLBuilder;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.LinearRegressionMetrics;
import com.datumbox.framework.core.machinelearning.preprocessing.OneHotEncoder;
import com.datumbox.framework.core.machinelearning.preprocessing.MinMaxScaler;
import com.datumbox.framework.core.machinelearning.regression.NLMS;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * DataModeling example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class DataModeling {
    
    /**
     * Example of how to use the Modeler class.
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
        Dataframe trainingDataframe;
        try (Reader fileReader = new InputStreamReader(new FileInputStream(Paths.get(Clustering.class.getClassLoader().getResource("datasets/labor-statistics/longley.csv").toURI()).toFile()), "UTF-8")) {
            LinkedHashMap<String, TypeInference.DataType> headerDataTypes = new LinkedHashMap<>();
            headerDataTypes.put("Employed", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("GNP.deflator", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("GNP", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Unemployed", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Armed.Forces", TypeInference.DataType.NUMERICAL);  
            headerDataTypes.put("Population", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("Year", TypeInference.DataType.NUMERICAL); 
            
            trainingDataframe = Dataframe.Builder.parseCSVFile(fileReader, "Employed", headerDataTypes, ',', '"', "\r\n", null, null, configuration);
        }
        catch(UncheckedIOException | IOException | URISyntaxException ex) {
            throw new RuntimeException(ex);
        }
        Dataframe testingDataframe = trainingDataframe.copy();
        
        
        
        //Setup Training Parameters
        //-------------------------
        Modeler.TrainingParameters trainingParameters = new Modeler.TrainingParameters();

        //numerical scaling configuration
        MinMaxScaler.TrainingParameters nsParams = new MinMaxScaler.TrainingParameters();
        trainingParameters.setNumericalScalerTrainingParameters(nsParams);

        //categorical encoding configuration
        OneHotEncoder.TrainingParameters ceParams = new OneHotEncoder.TrainingParameters();
        trainingParameters.setCategoricalEncoderTrainingParameters(ceParams);
        
        //Set feature selection configuration
        trainingParameters.setFeatureSelectorTrainingParametersList(Arrays.asList());

        //Model Configuration
        trainingParameters.setModelerTrainingParameters(new NLMS.TrainingParameters());
        
        
        
        //Fit the modeler
        //---------------
        Modeler modeler = MLBuilder.create(trainingParameters, configuration);
        modeler.fit(trainingDataframe);
        modeler.save("LaborStatistics");

        
        //Use the modeler
        //---------------

        //Make predictions on the test set
        modeler.predict(testingDataframe);

        LinearRegressionMetrics vm = new LinearRegressionMetrics(testingDataframe);
        
        System.out.println("Test Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Model Rsquare: "+vm.getRSquare());
        
        
        
        //Clean up
        //--------
        
        //Delete the modeler. This removes all files.
        modeler.delete();

        //Close Dataframes.
        trainingDataframe.close();
        testingDataframe.close();
    }
    
}

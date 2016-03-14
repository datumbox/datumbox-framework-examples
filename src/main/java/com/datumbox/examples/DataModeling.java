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

import com.datumbox.framework.applications.datamodeling.Modeler;
import com.datumbox.framework.common.Configuration;
import com.datumbox.framework.common.dataobjects.Dataframe;
import com.datumbox.framework.common.dataobjects.Record;
import com.datumbox.framework.common.dataobjects.TypeInference;
import com.datumbox.framework.common.utilities.PHPMethods;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.common.interfaces.ValidationMetrics;
import com.datumbox.framework.core.machinelearning.datatransformation.DummyXYMinMaxNormalizer;
import com.datumbox.framework.core.machinelearning.regression.NLMS;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Paths;
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
         * There are two configuration files in the resources folder:
         * 
         * - datumbox.config.properties: It contains the configuration for the storage engines (required)
         * - logback.xml: It contains the configuration file for the logger (optional)
         */
        
        //Initialization
        //--------------
        RandomGenerator.setGlobalSeed(42L); //optionally set a specific seed for all Random objects
        Configuration conf = Configuration.getConfiguration(); //default configuration based on properties file
        //conf.setDbConfig(new InMemoryConfiguration()); //use In-Memory storage (default)
        //conf.setDbConfig(new MapDBConfiguration()); //use MapDB storage
        //conf.getConcurrencyConfig().setParallelized(true); //turn on/off the parallelization
        //conf.getConcurrencyConfig().setMaxNumberOfThreadsPerTask(4); //set the concurrency level
        
        
        
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
            
            trainingDataframe = Dataframe.Builder.parseCSVFile(fileReader, "Employed", headerDataTypes, ',', '"', "\r\n", null, null, conf);
        }
        catch(UncheckedIOException | IOException | URISyntaxException ex) {
            throw new RuntimeException(ex);
        }
        Dataframe testingDataframe = trainingDataframe.copy();
        
        
        
        //Setup Training Parameters
        //-------------------------
        Modeler.TrainingParameters trainingParameters = new Modeler.TrainingParameters();
        
        //Model Configuration
        trainingParameters.setModelerClass(NLMS.class);
        trainingParameters.setModelerTrainingParameters(new NLMS.TrainingParameters());

        //Set data transfomation configuration
        trainingParameters.setDataTransformerClass(DummyXYMinMaxNormalizer.class);
        trainingParameters.setDataTransformerTrainingParameters(new DummyXYMinMaxNormalizer.TrainingParameters());
        
        //Set feature selection configuration
        trainingParameters.setFeatureSelectorClass(null);
        trainingParameters.setFeatureSelectorTrainingParameters(null);
        
        
        
        //Fit the modeler
        //---------------
        Modeler modeler = new Modeler("LaborStatistics", conf);
        modeler.fit(trainingDataframe, trainingParameters);
        
        
        
        //Use the modeler
        //---------------
        
        //Get validation metrics on the training set
        ValidationMetrics vm = modeler.validate(trainingDataframe);
        modeler.setValidationMetrics(vm); //store them in the model for future reference
        
        //Predict a new Dataframe
        modeler.predict(testingDataframe);
        
        System.out.println("Test Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Modeler Statistics: "+PHPMethods.var_export(vm));
        
        
        
        //Clean up
        //--------
        
        //Delete the modeler. This removes all files.
        modeler.delete();
        
        //Delete Dataframes.
        trainingDataframe.delete();
        testingDataframe.delete();
    }
    
}

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

import com.datumbox.applications.datamodeling.Modeler;
import com.datumbox.common.dataobjects.Dataset;
import com.datumbox.common.dataobjects.Record;
import com.datumbox.common.dataobjects.TypeInference;
import com.datumbox.common.persistentstorage.ConfigurationFactory;
import com.datumbox.common.persistentstorage.interfaces.DatabaseConfiguration;
import com.datumbox.common.utilities.PHPfunctions;
import com.datumbox.common.utilities.RandomGenerator;
import com.datumbox.framework.machinelearning.common.bases.mlmodels.BaseMLmodel;
import com.datumbox.framework.machinelearning.datatransformation.DummyXYMinMaxNormalizer;
import com.datumbox.framework.machinelearning.regression.NLMS;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Reader;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.HashMap;
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
     * @throws java.io.FileNotFoundException
     * @throws java.net.URISyntaxException
     */
    public static void main(String[] args) throws FileNotFoundException, URISyntaxException {      
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
        Reader fileReader = new FileReader(Paths.get(DataModeling.class.getClassLoader().getResource("datasets/labor-statistics/longley.csv").toURI()).toFile());

        Map<String, TypeInference.DataType> headerDataTypes = new HashMap<>();
        headerDataTypes.put("Employed", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("GNP.deflator", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("GNP", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("Unemployed", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("Armed.Forces", TypeInference.DataType.NUMERICAL);  
        headerDataTypes.put("Population", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("Year", TypeInference.DataType.NUMERICAL); 
        Dataset trainingDataset = Dataset.Builder.parseCSVFile(fileReader, "Employed", headerDataTypes, ',', '"', "\r\n", dbConf);
        Dataset testingDataset = trainingDataset.copy();
        
        
        
        //Setup Training Parameters
        //-------------------------
        Modeler.TrainingParameters trainingParameters = new Modeler.TrainingParameters();
        
        //Model Configuration
        trainingParameters.setMLmodelClass(NLMS.class);
        trainingParameters.setMLmodelTrainingParameters(new NLMS.TrainingParameters());

        //Set data transfomation configuration
        trainingParameters.setDataTransformerClass(DummyXYMinMaxNormalizer.class);
        trainingParameters.setDataTransformerTrainingParameters(new DummyXYMinMaxNormalizer.TrainingParameters());
        
        //Set feature selection configuration
        trainingParameters.setFeatureSelectionClass(null);
        trainingParameters.setFeatureSelectionTrainingParameters(null);
        
        
        
        //Fit the modeler
        //---------------
        Modeler modeler = new Modeler("LaborStatistics", dbConf);
        modeler.fit(trainingDataset, trainingParameters);
        
        
        
        //Use the modeler
        //---------------
        
        //Get validation metrics on the training set
        BaseMLmodel.ValidationMetrics vm = modeler.validate(trainingDataset);
        modeler.setValidationMetrics(vm); //store them in the model for future reference
        
        //Predict a new dataset
        modeler.predict(testingDataset);
        
        System.out.println("Test Results:");
        for(Integer rId: testingDataset) {
            Record r = testingDataset.get(rId);
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Modeler Statistics: "+PHPfunctions.var_export(vm));
        
        
        
        //Clean up
        //--------
        
        //Erase the modeler. This removes all files.
        modeler.erase();
        
        //Erase datasets.
        trainingDataset.erase();
        testingDataset.erase();
    }
    
}

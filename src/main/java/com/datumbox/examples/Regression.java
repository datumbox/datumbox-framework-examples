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

import com.datumbox.common.dataobjects.Dataset;
import com.datumbox.common.dataobjects.Record;
import com.datumbox.common.dataobjects.TypeInference;
import com.datumbox.common.persistentstorage.ConfigurationFactory;
import com.datumbox.common.persistentstorage.interfaces.DatabaseConfiguration;
import com.datumbox.common.utilities.PHPfunctions;
import com.datumbox.common.utilities.RandomGenerator;
import com.datumbox.framework.machinelearning.datatransformation.XYMinMaxNormalizer;
import com.datumbox.framework.machinelearning.featureselection.continuous.PCA;
import com.datumbox.framework.machinelearning.regression.MatrixLinearRegression;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Regression example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class Regression {
    
    /**
     * Example of how to use directly the algorithms of the framework in order to
     * perform regression. A similar approach can be used to perform clustering,
     * classification, build recommender system or perform topic modeling and dimensionality
     * reduction.
     * 
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.net.URISyntaxException
     */
    public static void main(String[] args) throws FileNotFoundException, URISyntaxException, IOException {   
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
        
        
        //Transform Dataset
        //-----------------
        
        //Normalize continuous variables
        XYMinMaxNormalizer dataTransformer = new XYMinMaxNormalizer("LaborStatistics", dbConf);
        dataTransformer.fit_transform(trainingDataset, new XYMinMaxNormalizer.TrainingParameters());
        


        //Feature Selection
        //-----------------
        
        //Perform dimensionality reduction using PCA
        
        PCA featureSelection = new PCA("LaborStatistics", dbConf);
        PCA.TrainingParameters featureSelectionParameters = new PCA.TrainingParameters();
        featureSelectionParameters.setMaxDimensions(trainingDataset.getVariableNumber()-1); //remove one dimension
        featureSelectionParameters.setWhitened(false);
        featureSelectionParameters.setVariancePercentageThreshold(0.99999995);
        featureSelection.fit_transform(trainingDataset, featureSelectionParameters);
        
        
        
        //Fit the regressor
        //-----------------
        
        MatrixLinearRegression regressor = new MatrixLinearRegression("LaborStatistics", dbConf);
        
        MatrixLinearRegression.TrainingParameters param = new MatrixLinearRegression.TrainingParameters();
        
        regressor.fit(trainingDataset, param);
        
        //Denormalize trainingDataset (optional)
        dataTransformer.denormalize(trainingDataset);
        
        
        
        //Use the regressor
        //------------------
        
        //Apply the same data transformations on testingDataset 
        dataTransformer.transform(testingDataset);
        
        //Apply the same featureSelection transformations on testingDataset
        featureSelection.transform(testingDataset);
        
        //Get validation metrics on the training set
        MatrixLinearRegression.ValidationMetrics vm = regressor.validate(testingDataset);
        regressor.setValidationMetrics(vm); //store them in the model for future reference
        
        //Denormalize testingDataset (optional)
        dataTransformer.denormalize(testingDataset);
        
        System.out.println("Results:");
        for(Integer rId: testingDataset) {
            Record r = testingDataset.get(rId);
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Regressor Statistics: "+PHPfunctions.var_export(vm));
        
        
        
        //Clean up
        //--------
        
        //Erase data transformer, featureselector and regressor.
        dataTransformer.erase();
        featureSelection.erase();
        regressor.erase();
        
        //Erase datasets.
        trainingDataset.erase();
        testingDataset.erase();
    }
    
}

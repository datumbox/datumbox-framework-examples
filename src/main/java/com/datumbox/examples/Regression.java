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
import com.datumbox.framework.core.machinelearning.datatransformation.XYMinMaxNormalizer;
import com.datumbox.framework.core.machinelearning.featureselection.continuous.PCA;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.LinearRegressionMetrics;
import com.datumbox.framework.core.machinelearning.regression.MatrixLinearRegression;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
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
        
        
        //Transform Dataframe
        //-----------------
        
        //Normalize continuous variables
        XYMinMaxNormalizer dataTransformer = MLBuilder.create(new XYMinMaxNormalizer.TrainingParameters(), conf);
        dataTransformer.fit_transform(trainingDataframe);
        dataTransformer.save("LaborStatistics");
        


        //Feature Selection
        //-----------------
        
        //Perform dimensionality reduction using PCA

        PCA.TrainingParameters featureSelectionParameters = new PCA.TrainingParameters();
        featureSelectionParameters.setMaxDimensions(trainingDataframe.xColumnSize()-1); //remove one dimension
        featureSelectionParameters.setWhitened(false);
        featureSelectionParameters.setVariancePercentageThreshold(0.99999995);

        PCA featureSelection = MLBuilder.create(featureSelectionParameters, conf);
        featureSelection.fit_transform(trainingDataframe);
        featureSelection.save("LaborStatistics");
        
        
        
        //Fit the regressor
        //-----------------

        MatrixLinearRegression.TrainingParameters param = new MatrixLinearRegression.TrainingParameters();

        MatrixLinearRegression regressor = MLBuilder.create(param, conf);
        regressor.fit(trainingDataframe);
        regressor.save("LaborStatistics");
        regressor.close(); //close the regressor, we will use it again later
        
        //Denormalize trainingDataframe (optional)
        dataTransformer.denormalize(trainingDataframe);


        
        //Use the regressor
        //------------------
        
        //Apply the same data transformations on testingDataframe 
        dataTransformer.transform(testingDataframe);
        
        //Apply the same featureSelection transformations on testingDataframe
        featureSelection.transform(testingDataframe);

        //Load again the regressor
        regressor = MLBuilder.load(MatrixLinearRegression.class, "LaborStatistics", conf);
        regressor.predict(testingDataframe);

        //Get validation metrics on the training set
        LinearRegressionMetrics vm = new LinearRegressionMetrics(testingDataframe);
        
        //Denormalize testingDataframe (optional)
        dataTransformer.denormalize(testingDataframe);
        
        System.out.println("Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Regressor Rsquare: "+vm.getRSquare());
        
        
        
        //Clean up
        //--------
        
        //Delete data transformer, featureselector and regressor.
        dataTransformer.delete();
        featureSelection.delete();
        regressor.delete();
        
        //Delete Dataframes.
        trainingDataframe.delete();
        testingDataframe.delete();
    }
    
}

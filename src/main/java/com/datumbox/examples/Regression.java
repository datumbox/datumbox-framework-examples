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
import com.datumbox.framework.core.machinelearning.featureselection.continuous.PCA;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.LinearRegressionMetrics;
import com.datumbox.framework.core.machinelearning.preprocessing.StandardScaler;
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


        //Transform Dataframe
        //-----------------

        //Scale continuous variables
        StandardScaler.TrainingParameters nsParams = new StandardScaler.TrainingParameters();
        nsParams.setScaleResponse(true);
        StandardScaler numericalScaler = MLBuilder.create(nsParams, configuration);

        numericalScaler.fit_transform(trainingDataframe);
        numericalScaler.save("LaborStatistics");



        //Feature Selection
        //-----------------
        
        //Perform dimensionality reduction using PCA

        PCA.TrainingParameters featureSelectionParameters = new PCA.TrainingParameters();
        featureSelectionParameters.setMaxDimensions(trainingDataframe.xColumnSize()-1); //remove one dimension
        featureSelectionParameters.setWhitened(false);
        featureSelectionParameters.setVariancePercentageThreshold(0.99999995);

        PCA featureSelection = MLBuilder.create(featureSelectionParameters, configuration);
        featureSelection.fit_transform(trainingDataframe);
        featureSelection.save("LaborStatistics");
        
        
        
        //Fit the regressor
        //-----------------

        MatrixLinearRegression.TrainingParameters param = new MatrixLinearRegression.TrainingParameters();

        MatrixLinearRegression regressor = MLBuilder.create(param, configuration);
        regressor.fit(trainingDataframe);
        regressor.save("LaborStatistics");
        regressor.close(); //close the regressor, we will use it again later


        
        //Use the regressor
        //------------------
        
        //Apply the same numerical scaling on testingDataframe
        numericalScaler.transform(testingDataframe);
        
        //Apply the same featureSelection transformations on testingDataframe
        featureSelection.transform(testingDataframe);

        //Load again the regressor
        regressor = MLBuilder.load(MatrixLinearRegression.class, "LaborStatistics", configuration);
        regressor.predict(testingDataframe);

        //Get validation metrics on the training set
        LinearRegressionMetrics vm = new LinearRegressionMetrics(testingDataframe);
        
        System.out.println("Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Regressor Rsquare: "+vm.getRSquare());
        
        
        
        //Clean up
        //--------
        
        //Delete scaler, featureselector and regressor.
        numericalScaler.delete();
        featureSelection.delete();
        regressor.delete();

        //Close Dataframes.
        trainingDataframe.close();
        testingDataframe.close();
    }
    
}

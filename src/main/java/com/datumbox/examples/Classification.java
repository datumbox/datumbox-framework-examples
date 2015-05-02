/**
 * Copyright (C) 2013-2015 Vasilis Vryniotis <bbriniotis@datumbox.com>
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
import com.datumbox.framework.machinelearning.classification.SoftMaxRegression;
import com.datumbox.framework.machinelearning.datatransformation.XMinMaxNormalizer;
import com.datumbox.framework.machinelearning.featureselection.continuous.PCA;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.HashMap;
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
        Reader fileReader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(Paths.get(Classification.class.getClassLoader().getResource("datasets/diabetes/diabetes.tsv.gz").toURI()).toFile()))));

        Map<String, TypeInference.DataType> headerDataTypes = new HashMap<>();
        headerDataTypes.put("pregnancies", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("plasma glucose", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("blood pressure", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("triceps thickness", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("serum insulin", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("bmi", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("dpf", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("age", TypeInference.DataType.NUMERICAL);
        headerDataTypes.put("test result", TypeInference.DataType.CATEGORICAL);
        
        
        Dataset trainingDataset = Dataset.Builder.parseCSVFile(fileReader, "test result", headerDataTypes, '\t', '"', "\r\n", dbConf);
        Dataset testingDataset = trainingDataset.copy();
        
        
        //Transform Dataset
        //-----------------
        
        //Normalize continuous variables
        XMinMaxNormalizer dataTransformer = new XMinMaxNormalizer("Diabetes", dbConf);
        dataTransformer.fit_transform(trainingDataset, new XMinMaxNormalizer.TrainingParameters());
        


        //Feature Selection
        //-----------------
        
        //Perform dimensionality reduction using PCA
        
        PCA featureSelection = new PCA("Diabetes", dbConf);
        PCA.TrainingParameters featureSelectionParameters = new PCA.TrainingParameters();
        featureSelectionParameters.setMaxDimensions(trainingDataset.getVariableNumber()-1); //remove one dimension
        featureSelectionParameters.setWhitened(false);
        featureSelectionParameters.setVariancePercentageThreshold(0.99999995);
        featureSelection.fit_transform(trainingDataset, featureSelectionParameters);
        
        
        
        //Fit the classifier
        //------------------
        
        SoftMaxRegression classifier = new SoftMaxRegression("Diabetes", dbConf);
        
        SoftMaxRegression.TrainingParameters param = new SoftMaxRegression.TrainingParameters();
        param.setTotalIterations(200);
        param.setLearningRate(0.1);
        
        classifier.fit(trainingDataset, param);
        
        //Denormalize trainingDataset (optional)
        dataTransformer.denormalize(trainingDataset);
        
        
        //Use the classifier
        //------------------
        
        //Apply the same data transformations on testingDataset 
        dataTransformer.transform(testingDataset);
        
        //Apply the same featureSelection transformations on testingDataset
        featureSelection.transform(testingDataset);
        
        //Get validation metrics on the training set
        SoftMaxRegression.ValidationMetrics vm = classifier.validate(testingDataset);
        classifier.setValidationMetrics(vm); //store them in the model for future reference
        
        //Denormalize testingDataset (optional)
        dataTransformer.denormalize(testingDataset);
        
        System.out.println("Results:");
        for(Integer rId: testingDataset) {
            Record r = testingDataset.get(rId);
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Classifier Statistics: "+PHPfunctions.var_export(vm));
        
        
        
        //Clean up
        //--------
        
        //Erase data transformer, featureselector and classifier.
        dataTransformer.erase();
        featureSelection.erase();
        classifier.erase();
        
        //Erase datasets.
        trainingDataset.erase();
        testingDataset.erase();
    }
    
}

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

import com.datumbox.framework.applications.nlp.TextClassifier;
import com.datumbox.framework.common.Configuration;
import com.datumbox.framework.core.common.dataobjects.Record;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.MLBuilder;
import com.datumbox.framework.core.machinelearning.classification.MultinomialNaiveBayes;
import com.datumbox.framework.core.machinelearning.featureselection.ChisquareSelect;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.ClassificationMetrics;
import com.datumbox.framework.core.common.text.extractors.NgramsExtractor;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


/**
 * Text Classification example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class TextClassification {
    
    /**
     * Example of how to use the TextClassifier class.
     * 
     * @param args the command line arguments
     * @throws java.net.URISyntaxException
     */
    public static void main(String[] args) throws URISyntaxException {        
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
        Map<Object, URI> datasets = new HashMap<>(); //The examples of each category are stored on the same file, one example per row.
        datasets.put("positive", TextClassification.class.getClassLoader().getResource("datasets/sentiment-analysis/rt-polarity.pos").toURI());
        datasets.put("negative", TextClassification.class.getClassLoader().getResource("datasets/sentiment-analysis/rt-polarity.neg").toURI());
        
        
        
        //Setup Training Parameters
        //-------------------------
        TextClassifier.TrainingParameters trainingParameters = new TextClassifier.TrainingParameters();

        //numerical scaling configuration
        trainingParameters.setNumericalScalerTrainingParameters(null);
        
        //Set feature selection configuration
        trainingParameters.setFeatureSelectorTrainingParametersList(Arrays.asList(new ChisquareSelect.TrainingParameters()));
        
        //Set text extraction configuration
        trainingParameters.setTextExtractorParameters(new NgramsExtractor.Parameters());

        //Classifier configuration
        trainingParameters.setModelerTrainingParameters(new MultinomialNaiveBayes.TrainingParameters());
        
        
        
        //Fit the classifier
        //------------------
        TextClassifier textClassifier = MLBuilder.create(trainingParameters, configuration);
        textClassifier.fit(datasets);
        textClassifier.save("SentimentAnalysis");
        
        
        
        //Use the classifier
        //------------------
        
        //Get validation metrics on the dataset
        ClassificationMetrics vm = textClassifier.validate(datasets);
        
        //Classify a single sentence
        String sentence = "Datumbox is amazing!";
        Record r = textClassifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
        System.out.println("Probability: "+r.getYPredictedProbabilities().get(r.getYPredicted()));
        
        System.out.println("Classifier Accuracy: "+vm.getAccuracy());
        
        
        
        //Clean up
        //--------
        
        //Delete the classifier. This removes all files.
        textClassifier.delete();
    }
    
}

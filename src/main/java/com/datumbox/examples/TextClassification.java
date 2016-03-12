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

import com.datumbox.framework.applications.nlp.TextClassifier;
import com.datumbox.framework.common.Configuration;
import com.datumbox.framework.common.dataobjects.Record;
import com.datumbox.framework.common.utilities.PHPMethods;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.classification.MultinomialNaiveBayes;
import com.datumbox.framework.core.machinelearning.common.interfaces.ValidationMetrics;
import com.datumbox.framework.core.machinelearning.featureselection.categorical.ChisquareSelect;
import com.datumbox.framework.core.utilities.text.extractors.NgramsExtractor;
import java.net.URI;
import java.net.URISyntaxException;
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
        Map<Object, URI> Dataframe = new HashMap<>(); //The examples of each category are stored on the same file, one example per row.
        Dataframe.put("positive", TextClassification.class.getClassLoader().getResource("datasets/sentiment-analysis/rt-polarity.pos").toURI());
        Dataframe.put("negative", TextClassification.class.getClassLoader().getResource("datasets/sentiment-analysis/rt-polarity.neg").toURI());
        
        
        
        //Setup Training Parameters
        //-------------------------
        TextClassifier.TrainingParameters trainingParameters = new TextClassifier.TrainingParameters();
        
        //Classifier configuration
        trainingParameters.setModelerClass(MultinomialNaiveBayes.class);
        trainingParameters.setModelerTrainingParameters(new MultinomialNaiveBayes.TrainingParameters());
        
        //Set data transfomation configuration
        trainingParameters.setDataTransformerClass(null);
        trainingParameters.setDataTransformerTrainingParameters(null);
        
        //Set feature selection configuration
        trainingParameters.setFeatureSelectorClass(ChisquareSelect.class);
        trainingParameters.setFeatureSelectorTrainingParameters(new ChisquareSelect.TrainingParameters());
        
        //Set text extraction configuration
        trainingParameters.setTextExtractorClass(NgramsExtractor.class);
        trainingParameters.setTextExtractorParameters(new NgramsExtractor.Parameters());
        
        
        
        //Fit the classifier
        //------------------
        TextClassifier classifier = new TextClassifier("SentimentAnalysis", conf);
        classifier.fit(Dataframe, trainingParameters);
        
        
        
        //Use the classifier
        //------------------
        
        //Get validation metrics on the training set
        ValidationMetrics vm = classifier.validate(Dataframe);
        classifier.setValidationMetrics(vm); //store them in the model for future reference
        
        //Classify a single sentence
        String sentence = "Datumbox is amazing!";
        Record r = classifier.predict(sentence);
        
        System.out.println("Classifing sentence: \""+sentence+"\"");
        System.out.println("Predicted class: "+r.getYPredicted());
        System.out.println("Probability: "+r.getYPredictedProbabilities().get(r.getYPredicted()));
        
        System.out.println("Classifier Statistics: "+PHPMethods.var_export(vm));
        
        
        
        //Clean up
        //--------
        
        //Delete the classifier. This removes all files.
        classifier.delete();
    }
    
}

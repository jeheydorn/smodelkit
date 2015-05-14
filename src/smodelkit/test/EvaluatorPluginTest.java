package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.Evaluation;
import smodelkit.MLSystemsManager;
import smodelkit.evaluator.AccuracyPerColumn;
import smodelkit.evaluator.TopN;

public class EvaluatorPluginTest
{

	@Test
	public void loadEvaluatorTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation = new MLSystemsManager().run("-L smodelkit.learner.NeuralNet model_settings/neuralnet_test.json -A Datasets/mcc/iris.arff -E random 0.1 -M smodelkit.evaluator.TopN 1 -R 0".split(" "), null);
		assertEquals(0.9333333333333333, evaluation.getScores(TopN.class).get(0), 0.000000001);	
	}


	@Test
	public void loadEvaluatorWithMutlipleArgsTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation = new MLSystemsManager().run("-L ic model_settings/ic.json -A Datasets/mdc/synthetic/continuous_2out_4class.arff -E random 0.5 -M smodelkit.evaluator.TopN 1 2 3 -R 0 --rows 500 --threads 1".split(" "), null);
		assertEquals(0.148, evaluation.getScores(TopN.class).get(0), 0.000000001);	
		assertEquals(0.228, evaluation.getScores(TopN.class).get(1), 0.000000001);	
		assertEquals(0.372, evaluation.getScores(TopN.class).get(2), 0.000000001);	
	}

	@Test
	public void multipleEvaluatorTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation = new MLSystemsManager().run("-L ic model_settings/ic.json -A Datasets/mdc/synthetic/continuous_2out_4class.arff -E random 0.5 -M smodelkit.evaluator.TopN 1 2 3 end percolumn -R 0 --rows 500 --threads 1".split(" "), null);
		assertEquals(0.148, evaluation.getScores(TopN.class).get(0), 0.000000001);	
		assertEquals(0.228, evaluation.getScores(TopN.class).get(1), 0.000000001);	
		assertEquals(0.372, evaluation.getScores(TopN.class).get(2), 0.000000001);	
		assertEquals(0.148, evaluation.getScores(AccuracyPerColumn.class).get(0), 0.412);	
		assertEquals(0.148, evaluation.getScores(AccuracyPerColumn.class).get(0), 0.336);	
	}

}

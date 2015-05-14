package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.Evaluation;
import smodelkit.MLSystemsManager;
import smodelkit.evaluator.AccuracyOfGroup;

public class AccuracyOfGroupTest {

	@Test
	public void test() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException 
	{
		Evaluation evaluation = new MLSystemsManager().run(("-L smodelkit.learner.NeuralNet model_settings/neuralnet_test.json -A"
				+ " Datasets/mdc/synthetic/continuous_2out_4class.arff -E random 0.5 -M accuracy-group class1 class2 --rows 300 -R 0").split(" "), null);
		assertEquals(0.41333333333333333, evaluation.getScores(AccuracyOfGroup.class).get(0), 0.000000001);	
	}

}

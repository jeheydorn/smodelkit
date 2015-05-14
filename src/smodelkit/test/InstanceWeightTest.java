package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.Evaluation;
import smodelkit.MLSystemsManager;
import smodelkit.evaluator.TopN;

public class InstanceWeightTest
{

	@Test
	public void neuralNetTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{
		Evaluation evaluation = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/test/weighted_instances.arff -E training -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.3333333, evaluation.getScores(TopN.class).get(0), 0.0001);
	}

	/**
	 * Makes sure that reordering outputs doesn't remove instance weights.
	 */
	@Test
	public void neuralNetReorderOutputsTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{
		Evaluation evaluation = new MLSystemsManager().run("-L neuralnet model_settings/neuralnet_test.json -A Datasets/test/weighted_instances.arff -E training -M top-n 1 -C class2 class1 -R 0".split(" "), null);
		assertEquals(0.3333333, evaluation.getScores(TopN.class).get(0), 0.0001);
	}

	@Test
	public void rankedccTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{
		Evaluation evaluation = new MLSystemsManager().run("-L rankedcc model_settings/rankedcc.json -A Datasets/test/weighted_instances.arff -E training -M top-n 1 -R 0".split(" "), null);
		assertEquals(0.3333333, evaluation.getScores(TopN.class).get(0), 0.0001);
	}

}

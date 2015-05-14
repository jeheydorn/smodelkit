package smodelkit.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import org.junit.Test;

import smodelkit.Evaluation;
import smodelkit.MLSystemsManager;
import smodelkit.evaluator.TopN;

public class RankedCCTest
{

	@Test
	public void test() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{
		Evaluation evaluation = new MLSystemsManager().run("-L rankedcc model_settings/rankedcc.json -A Datasets/mdc/synthetic/nominal_4classes.arff -E random 0.1 -M top-n 1 --rows 200 -R 0".split(" "), null);
		assertEquals(0.45, evaluation.getScores(TopN.class).get(0), 0.000000001);
	}

	@Test
	public void missingLabelsTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException
	{
		try
		{
			Evaluation evaluation = new MLSystemsManager().run("-L rankedcc model_settings/rankedcc.json -A Datasets/mdc/bridges_original.arff -E random 0.1 -M top-n 1 -R 0".split(" "), null);
			assertEquals(0.45, evaluation.getScores(TopN.class).get(0), 0.000000001);
		}
		catch (IllegalArgumentException e)
		{
			assertEquals("RankedCC cannot handle unknown outputs.", e.getMessage());
			return;
		}
		fail();
	}

}

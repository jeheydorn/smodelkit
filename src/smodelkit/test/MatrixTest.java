package smodelkit.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Range;

public class MatrixTest
{
	@Test
	public void selectColumnsTest()
	{
		Matrix m = new Matrix();
		m.loadFromArffString("% continuousInputs: false\n" + 
				"% addNoisyInputs: false\n" + 
				"% inputNoiseProbability: 0.1\n" + 
				"% outputNoiseProbability: 0.1\n" + 
				"% generator type: class smodelkit.SyntheticDataGenerator$LimitedOutputsGenerator\n" + 
				"% numOutputVectorsPerInput: 3\n" + 
				"% resultRows: 10000\n" + 
				"% Number of inputs before adding noise (numInputsVectors): 20\n" + 
				"% numOutputs = 3\n" + 
				"% numOutputVectors = 20\n" + 
				"@RELATION 'synthetic_data -c -3'\n" + 
				"@ATTRIBUTE x1 {a, b, c, d}\n" + 
				"@ATTRIBUTE x2 {a, b, c, d}\n" + 
				"@ATTRIBUTE x3 {a, b, c, d}\n" + 
				"@ATTRIBUTE class1 {e, f, g, h}\n" + 
				"@ATTRIBUTE class2 {e, f, g, h}\n" + 
				"@ATTRIBUTE class3 {e, f, g, h}\n" + 
				"@DATA\n" + 
				"b, d, b, g, e, f\n" + 
				"a, c, c, h, f, f\n" + 
				"c, b, d, e, e, g\n" + 
				"c, b, b, g, e, f\n" + 
				"a, b, c, g, e, g\n" + 
				"b, a, d, g, h, g\n" + 
				"");
		
		{
			Matrix selected = m.selectColumns(Arrays.asList(0, 1, 2));
			assertEquals("x1", selected.getAttrName(0));
			assertEquals("x2", selected.getAttrName(1));
			assertEquals("x3", selected.getAttrName(2));
			assertEquals(m.rows(), selected.rows());
		}
		{
			Matrix selected = m.selectColumns(Arrays.asList(2, 0, 1));
			assertEquals("x3", selected.getAttrName(0));
			assertEquals("x1", selected.getAttrName(1));
			assertEquals("x2", selected.getAttrName(2));
			assertEquals(m.rows(), selected.rows());
		}
		{
			Matrix selected = m.selectColumns(Arrays.asList(0, 1, 2, 5, 4));
			assertEquals("x1", selected.getAttrName(0));
			assertEquals("x2", selected.getAttrName(1));
			assertEquals("x3", selected.getAttrName(2));
			assertEquals("class3", selected.getAttrName(3));
			assertEquals("class2", selected.getAttrName(4));
			assertEquals(m.rows(), selected.rows());
		}
	}
	
	@Test
	public void keyValuePairsPositiveTest()
	{
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	{pig,sheep}\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"{0 white,1 sheep, 2 high}\n"); 
			assertEquals(1, data.row(0).get(0), 0.000000001);
			assertEquals(1, data.row(0).get(1), 0.000000001);
			assertEquals(2, data.row(0).get(2), 0.000000001);
		}

		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	continuous\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"{0 white,1 100, 2 high}\n"); 
			assertEquals(1, data.row(0).get(0), 0.000000001);
			assertEquals(100, data.row(0).get(1), 0.000000001);
			assertEquals(2, data.row(0).get(2), 0.000000001);
		}

		// Change order in which attribute values are specified.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	continuous\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"{1 100, 2 high, 0 white}\n"); 
			assertEquals(1, data.row(0).get(0), 0.000000001);
			assertEquals(100, data.row(0).get(1), 0.000000001);
			assertEquals(2, data.row(0).get(2), 0.000000001);
	
		}
		
		// Ommit values
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	continuous\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"{0 white}\n"); 
			assertEquals(1, data.row(0).get(0), 0.000000001);
			assertEquals(0, data.row(0).get(1), 0.000000001);
			assertEquals(0, data.row(0).get(2), 0.000000001);
		}

		// Make sure an empty line is not counted as an instance.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	continuous\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"\n"); 
			assertEquals(0, data.rows());
		}
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	continuous\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"\n" +
					"{0 white}\n\n\n");
			assertEquals(1, data.rows());
		}

		// Empty instance.
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION simple_nominal\n" + 
					"\n" + 
					"@ATTRIBUTE x1	{pink,white,black}\n" + 
					"@ATTRIBUTE x2	continuous\n" + 
					"@ATTRIBUTE class2	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"{}");
			assertEquals(1, data.rows());
			assertEquals(0, data.row(0).get(0), 0.000000001);
			assertEquals(0, data.row(0).get(1), 0.000000001);
			assertEquals(0, data.row(0).get(2), 0.000000001);
		}

	}
	
	@Test(expected = IllegalArgumentException.class)
	public void keyValuePairsBadAttributeValueTest1()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{pink,white,black}\n" + 
				"@ATTRIBUTE x2	continuous\n" + 
				"@ATTRIBUTE class2	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"{0 1,1 100, 2 high}\n"); 
		assertEquals(1, data.row(0).get(0), 0.000000001);
		assertEquals(100, data.row(0).get(1), 0.000000001);
		assertEquals(2, data.row(0).get(2), 0.000000001);
	}

	@Test(expected = IllegalArgumentException.class)
	public void keyValuePairsBadAttributeValueTest2()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{pink,white,black}\n" + 
				"@ATTRIBUTE x2	continuous\n" + 
				"@ATTRIBUTE class2	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"{0 black,1 pink, 2 high}\n"); 
		assertEquals(1, data.row(0).get(0), 0.000000001);
		assertEquals(100, data.row(0).get(1), 0.000000001);
		assertEquals(2, data.row(0).get(2), 0.000000001);
	}

	@Test(expected = IllegalArgumentException.class)
	public void keyValuePairsBadAttributeValueTest3()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{pink,white,black}\n" + 
				"@ATTRIBUTE x2	continuous\n" + 
				"@ATTRIBUTE class2	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"{10 black,1 100, 2 high}\n"); 
		assertEquals(1, data.row(0).get(0), 0.000000001);
		assertEquals(100, data.row(0).get(1), 0.000000001);
		assertEquals(2, data.row(0).get(2), 0.000000001);
	}
	
	@Test
	public void toStringSparseTest()
	{
		Matrix irisNotSparse = new Matrix();
		irisNotSparse.loadFromArffFile("Datasets/mcc/iris.arff");
		Matrix irisSparse = new Matrix();
		String sparseStr = irisNotSparse.toStringSparse();
		irisSparse.loadFromArffString(sparseStr);
		for (int r : new Range(irisSparse.rows()))
		{
			assertEquals(irisNotSparse.row(r), irisSparse.row(r));
		}
	}

	@Test
	public void toStringTest()
	{
		Matrix iris = new Matrix();
		iris.loadFromArffFile("Datasets/mcc/iris.arff");
		Matrix irisFromString = new Matrix();
		irisFromString.loadFromArffString(iris.toString());
		for (int r : new Range(irisFromString.rows()))
		{
			assertEquals(iris.row(r), irisFromString.row(r));
		}
	}
	
	@Test
	public void addEmptyColumnTest()
	{
		Matrix m = new Matrix();
		m.addEmptyColumn("empty_column1");
		assertEquals("empty_column1", m.getAttrName(0));
		assertEquals(1, m.cols());
		assertTrue(m.isContinuous(0));

		m.addEmptyColumn("empty_column2");
		assertEquals("empty_column2", m.getAttrName(1));
		assertEquals(2, m.cols());
		assertTrue(m.isContinuous(1));
		
		m.addAttributeValue(1, "value1");
		assertFalse(m.isContinuous(1));
		m.addAttributeValue(1, "value2");
		assertFalse(m.isContinuous(1));
		m.addAttributeValue(1, "value3");
		assertFalse(m.isContinuous(1));
		assertTrue(m.isContinuous(0));
		assertEquals("value1", m.getAttrValueName(1, 0));
		assertEquals("value2", m.getAttrValueName(1, 1));
		assertEquals("value3", m.getAttrValueName(1, 2));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void addEmptyColumnNegativeTest()
	{
		Matrix m = new Matrix();
		m.addEmptyColumn("duplicate_column");
		m.addEmptyColumn("duplicate_column");		
	}

	@Test(expected=IllegalArgumentException.class)
	public void addAttributeValueNegativeTest()
	{
		Matrix m = new Matrix();
		m.addEmptyColumn("empty_column");
		m.addAttributeValue(0, "duplicate");
		m.addAttributeValue(0, "duplicate");		
	}
	
	@Test
	public void instanceWeightTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med\n" + 
				"0.1,0.2,low\n" + 
				"0.9,0.3,high\n" + 
				"0.9,0.3,high\n" + 
				"");
		for (int r : new Range(data.rows()))
		{
			assertEquals(1.0, data.row(r).getWeight(), Double.MIN_VALUE);
		}
		
		data.row(0).setWeight(2.0);
		assertEquals(2.0, data.row(0).getWeight(), Double.MIN_VALUE);
		for (int r : new Range(1, data.rows()))
		{
			assertEquals(1.0, data.row(r).getWeight(), Double.MIN_VALUE);
		}
		
		data.row(2).setWeight(0.0);
		assertEquals(0.0, data.row(2).getWeight(), Double.MIN_VALUE);
		assertEquals(2.0, data.row(0).getWeight(), Double.MAX_VALUE);
		
		// Add a new row and make sure the instance weight is correct.
		data.addRow(new Vector(new double[] {0.1, 0.2, 0.0}));
		assertEquals(1.0, data.row(data.rows() - 1).getWeight(), Double.MIN_VALUE);
		
		data.addRow(new Vector(new double[] {0.1, 0.2, 2.0}, 10.0));
		assertEquals(10.0, data.row(data.rows() - 1).getWeight(), Double.MIN_VALUE);
		
		
	}
		
	@Test(expected=IllegalArgumentException.class)
	public void instanceWeightNegativeTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med\n" + 
				"0.1,0.2,low\n" + 
				"0.9,0.3,high\n" + 
				"0.9,0.3,high\n" + 
				"");

		data.row(0).setWeight(-1.0);
	}
	
	@Test
	public void iteratorTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med\n" + 
				"0.1,0.2,low\n" + 
				"0.9,0.3,high\n" + 
				"");
		
		List<double[]> expected = Arrays.asList(
				new double[] {0.4, 0.9, 1.0},
				new double[] {0.1, 0.2, 0.0}, 
				new double[] {0.9, 0.3, 2.0});
		
		Iterator<Vector> it = data.iterator();
		int i = 0;
		while (it.hasNext())
		{
			Vector actual = it.next();
			for (int c : new Range(expected.get(i).length))
				assertEquals(expected.get(i)[c], actual.get(c), Double.MIN_VALUE);
			i++;
		}
		assertFalse(it.hasNext());
	}
	
	@Test
	public void emptyIteratorTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"");
		for (@SuppressWarnings("unused") Vector row : data)
		{
			// This should not be reached.
			fail();
		}
	}
	
	@Test
	public void parseInstanceWeightTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med, {2}\n" + 
				"0.1,0.2,low,{1}\n" + 
				"0.9,0.3,high\n" + 
				"0.8,0.7,high,{ 4.02 }\n" + 
				"");
		
		assertEquals(2.0, data.row(0).getWeight(), Double.MIN_VALUE);
		assertEquals(1.0, data.row(1).getWeight(), Double.MIN_VALUE);
		assertEquals(1.0, data.row(2).getWeight(), Double.MIN_VALUE);
		assertEquals(4.02, data.row(3).getWeight(), Double.MIN_VALUE);
	}

	@Test
	public void parseInstanceWeightSparseTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"{0 0.4,1 0.9,2 med}, {2}\n" + 
				"{0 0.1,1 0.2,2 low},{1}\n" + 
				"{0 0.9,1 0.3,2 high}\n" + 
				"{0 0.8,1 0.7,2 high},{ 4.02 }\n" + 
				"");
		
		assertEquals(2.0, data.row(0).getWeight(), Double.MIN_VALUE);
		assertEquals(1.0, data.row(1).getWeight(), Double.MIN_VALUE);
		assertEquals(1.0, data.row(2).getWeight(), Double.MIN_VALUE);
		assertEquals(4.02, data.row(3).getWeight(), Double.MIN_VALUE);
	}

	@Test(expected=IllegalArgumentException.class)
	public void parseNegativeInstanceWeightTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med, {2}\n" + 
				"0.1,0.2,low,{-1.01}" + 
				"");
	}

	@Test(expected=NumberFormatException.class)
	public void parseInstanceBadNumberTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.1,0.2,low,{-1.01a}" + 
				"");
	}

	@Test
	public void saveInstanceWeightsTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	real\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med\n" + 
				"0.4,0.9,med\n" + 
				"");		
		data.row(0).setWeight(0.1);
		assertEquals("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	NUMERIC\n" + 
				"@ATTRIBUTE x2	NUMERIC\n" + 
				"@ATTRIBUTE class1	{low, med, high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med, {0.1}\n" + 
				"0.4,0.9,med\n",
				data.toString());		
	}

	@Test
	public void saveInstanceWeightsSparseTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	NUMERIC\n" + 
				"@ATTRIBUTE x2	NUMERIC\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med\n" + 
				"0.4,0.9,med\n" + 
				"");		
		data.row(0).setWeight(0.1);
		assertEquals("@RELATION simple_nominal\n" + 
				"\n" + 
				"@ATTRIBUTE x1	NUMERIC\n" + 
				"@ATTRIBUTE x2	NUMERIC\n" + 
				"@ATTRIBUTE class1	{low, med, high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"{0 0.4,1 0.9,2 med}, {0.1}\n" +
				"{0 0.4,1 0.9,2 med}\n", 
				data.toStringSparse());		
	}
	
	@Test
	public void loadDatasetWithSpaceInRelationNameTest()
	{
		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION \"name with spaces\"\n" + 
					"\n" + 
					"@ATTRIBUTE x1	NUMERIC\n" + 
					"@ATTRIBUTE x2	NUMERIC\n" + 
					"@ATTRIBUTE class1	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"0.4,0.9,med\n" + 
					"");
			assertEquals("name with spaces", data.getRelationName());
		}

		{
			Matrix data = new Matrix();
			data.loadFromArffString("@RELATION 'name with spaces'\n" + 
					"\n" + 
					"@ATTRIBUTE x1	NUMERIC\n" + 
					"@ATTRIBUTE x2	NUMERIC\n" + 
					"@ATTRIBUTE class1	{low,med,high}\n" + 
					"\n" + 
					"@DATA\n" + 
					"0.4,0.9,med\n" + 
					"");
			assertEquals("name with spaces", data.getRelationName());
		}
	}
	
	@Test
	public void colonInRelationNameTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION \"aName: -c -2 \"\n" + 
				"\n" + 
				"@ATTRIBUTE x1	NUMERIC\n" + 
				"@ATTRIBUTE x2	NUMERIC\n" + 
				"@ATTRIBUTE class1	{low,med,high}\n" + 
				"@ATTRIBUTE class2	{low,med,high}\n" + 
				"\n" + 
				"@DATA\n" + 
				"0.4,0.9,med\n" + 
				"");
		assertTrue(data.toString().contains(":"));
		
	}

}

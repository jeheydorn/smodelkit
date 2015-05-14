package smodelkit.test;

//import static java.lang.System.out;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import smodelkit.filter.NominalToCategorical;
import smodelkit.filter.Normalize;


public class FilterTest
{
	@Test
	public void TestFindFilter()
	{
		NominalToCategorical nomToCat = new NominalToCategorical();
		Normalize normalize = new Normalize();
		normalize.setInnerFilter(nomToCat);
		assertEquals(normalize, normalize.findFilter(Normalize.class));
		assertEquals(nomToCat, normalize.findFilter(NominalToCategorical.class));
		assertEquals(nomToCat, nomToCat.findFilter(NominalToCategorical.class));
		assertEquals(null, nomToCat.findFilter(Normalize.class));		
	}

	@Test
	public void TestIncludes()
	{
		NominalToCategorical nomToCat = new NominalToCategorical();
		Normalize normalize = new Normalize();
		normalize.setInnerFilter(nomToCat);
		assertTrue(normalize.includes(Normalize.class));
		assertTrue(normalize.includes(NominalToCategorical.class));
		assertTrue(nomToCat.includes(NominalToCategorical.class));
		assertFalse(nomToCat.includes(Normalize.class));		
		
	}
}

package smodelkit.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import smodelkit.QuoteParser;
import smodelkit.util.Helper;

public class QuoteParserTest
{

	@Test
	public void test()
	{
		{
			QuoteParser parser = new QuoteParser("@attribute 'departwb'   numeric");
			assertTrue(parser.hasNext());
			assertEquals("@attribute", parser.next());
			assertEquals("'departwb'", parser.next());
			assertEquals("numeric", parser.next());
			assertFalse(parser.hasNext());
		}
		{
			QuoteParser parser = new QuoteParser("@attribute 'depar  twb'   numeric");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("@attribute", "'depar  twb'", "numeric"), tokens);
		}
		{
			QuoteParser parser = new QuoteParser("@attribute '\\'aUAm inp2\\'l1' numeric");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("@attribute", "'\\'aUAm inp2\\'l1'", "numeric"), tokens);
		}
		{
			QuoteParser parser = new QuoteParser("@attribute '\\'aUAminp2\\'l1' numeric");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("@attribute", "'\\'aUAminp2\\'l1'", "numeric"), tokens);
		}
		{
			QuoteParser parser = new QuoteParser("'\\'(267.5-268.5]\\''");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("'\\'(267.5-268.5]\\''"), tokens);
		}
		{
			QuoteParser parser = new QuoteParser(" This is a sentence which has no quotes.   ");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("This", "is", "a", "sentence", "which", "has", "no", "quotes."),
					tokens);
		}
		
		// Repeat the above tests but with double quotes.
		{
			QuoteParser parser = new QuoteParser("@attribute \"departwb\"   numeric");
			assertTrue(parser.hasNext());
			assertEquals("@attribute", parser.next());
			assertEquals("\"departwb\"", parser.next());
			assertEquals("numeric", parser.next());
			assertFalse(parser.hasNext());
		}
		{
			QuoteParser parser = new QuoteParser("@attribute \"\\\"aUAminp2\\\"l1\" numeric");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("@attribute", "\"\\\"aUAminp2\\\"l1\"", "numeric"), tokens);
		}
		{
			QuoteParser parser = new QuoteParser("\"\\\"(267.5-268.5]\\\"\"");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("\"\\\"(267.5-268.5]\\\"\""), tokens);
		}
		{
			QuoteParser parser = new QuoteParser(" This is a sentence which has no quotes.   ");
			List<String> tokens = Helper.iteratorToList(parser);
			assertEquals(Arrays.asList("This", "is", "a", "sentence", "which", "has", "no", "quotes."),
					tokens);
		}
		
		{
			QuoteParser parser = new QuoteParser("-M 2");
			assertTrue(parser.hasNext());
			assertEquals("-M", parser.next());
			assertTrue(parser.hasNext());
			assertEquals("2", parser.next());
			assertFalse(parser.hasNext());
			
			
		}
	}

}

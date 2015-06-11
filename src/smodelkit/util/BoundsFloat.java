package smodelkit.util;

import java.io.Serializable;

@SuppressWarnings("serial")
public class BoundsFloat implements Serializable
{
	public float lower;
    public float upper;
    
    public BoundsFloat()
    {
    	lower = Float.NEGATIVE_INFINITY;
    	upper = Float.POSITIVE_INFINITY;
    }
    
    public BoundsFloat(float lower, float upper)
    {
    	this.lower = lower;
    	this.upper = upper;
    }
    
    @Override
    public String toString()
    {
    	return (Float.isFinite(lower) ? "(" : "[") + lower + ", " + upper + 
    			(Float.isFinite(upper) ? ")" : "]");
    }
    
    public float getSpan()
    {
    	return upper - lower;
    }
}

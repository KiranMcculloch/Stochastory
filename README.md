# Stochastory
Randomly generated world maps from an alternate universe.

### AI Disclaimer
Yes, AI was used in the making of this program, specifically [] hosted by the University of Edinburgh's ELM. It helped write code but is also an integral part of the workings of the program, as it is capable of dynamically generating names and historical details in natural language out of datapoints alone (although they can still sound a bit generic.) Absolutely no AI was used for writing this description. 

### How to use
Download the code, and run main.py in the terminal or run main.sh (which does the same thing but can be treated like an executable on some systems.) If you select **Online Mode**, you'll need an ELM API key for it to work, which you can get here: 

If you don't have an API key, you can still generate the geojson map or provinces, with colors and the data that would have been used for generating name, but it will be unflavored and unstyled.

Put your API key into the GUI and hit start - depending on your internet speed, it may take more or less time, but I found a typical run is about 5-6 minutes, where the bottleneck is the LLM steps, as networking takes a while for that many features.

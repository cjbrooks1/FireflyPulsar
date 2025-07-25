# FireflyPulsar

This Firefly_Pulsar Generator can be used to create a pulse signal that is similar in structure to a pulsar's signal but is optimized to be maximally distinct in terms of its structure such as its period and absolute duty cycle ( the amount of time during a pulsar' period that its pulse is greater than the user defined flux cut off) while taking into account the potential energy requirements of generating such a pulse. This is managed using a weight system where #ws represents the weight assigned to how important dissimilarity should be considered by the model. A higher ws value means more significance is given to disimilarity and less significance is given to reducing potential energy usage.

# Using the Model

To use the model, run the firefly_pulsar_generator in the folder/ directory where you want the data and figures to be saved to. 

The model will prompt you regarding parameters it needs to find and generate the pulsar background(if you do not have one already in a .txt file)
###Important to note is that the X,Y,Z coordinates are defined with Earth being at 0,0,0

It will then ask you for more information regarding the ws weight(s) you wish to use when generating the Firefly Pulse Signal.
###If you decide to search for multiple weights, list the weights seperated by a coma and no space. No need to include square brackets or parenthesis.

Next, it will  prompt you for additional preferences such as whether you want to generate contour or heat maps and whether you want to save the generator's  pulse signal and cost search landscape.
Saving the pulse signal and cost search landscape allows you to see the data behind the matrix that is used to find the most optimized (lowest cost) Firefly Pulse Signal and is the basis of the heat and contour map.

###The files will be saved as .txt files  and .png files within the directory that the code is executed in.



Welcome to the un-spot-able github!  

Unfortunately starry is no longer maintained, so it can be complicated to get it working (even the modified version found here).

The best way to get this code running on your own device is as follows:

1. Have Python 3.10 available on your computer.  You can download it directly, but I recommend using pyenv to avoid messing up your computer's python environment.

2. Download/Clone this repository onto your local device, ensuring to keep the file structure the same

3. Use your preferred code editor (I prefer VS Code) to open the repository and create a virtual enviorment in 3.10.
NOTE: I recommend hard installing the requirements.txt file for this code, so your initial virtual enviorment should be empty.

4. Once your enviorment is set up, run the following command in terminal (checking you are in the correct enviorment):
'pip install -r requirements.txt --no-deps'
Unfortunately this code only support pip and has not been tested with conda.  Sorry conda users :(

Once everything is installed, we can start using the code!

1. To run everything, type "python unspotable.py eridani.cfg" into the terminal while in the unspotable directory

3. If everything is working, the code should start running and produce all the plots found in eridani-outputs/quad-limb-70i-units
NOTE: the above command will overwrite the files found in eridani-outputs/quad-limb-70i-units unless eridani.cfg is changed.

4. From there it is time to explore the code more.  I'd recommend playing around with a new .cfg file using your own stellar parameters.  Just be sure to change the outdir or folder before running.

5. There are also a handful of notebooks in the notebooks folder that show some of the cool things you can do with the results of the code.

Now go enjoy creating unspotable patterns and reach out with any question!

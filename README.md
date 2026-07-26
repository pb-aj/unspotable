Welcome to the un-spot-able github!  

Unfortunately starry is no longer maintained, so it can be complicated to get it working (even the modified version found here).

The best way to get this code running on your own device is as follows:

1. Have Python 3.10 available on your computer.  You can download it directly, but I recommend using pyenv to avoid messing up your computer's python.

2. Download/Clone this repository onto your local device, ensuring to keep the file structure the same

3. Use your preferred code editor (I prefer VS Code) to open the repository and create a virtual enviorment in 3.10.
NOTE: I recommend hard installing the requirements.txt file for this code, so your initial virtual enviorment should be empty

4. Once your enviorment is set up, run the following command in terminal (checking you are in the correct enviorment):
'pip install -r requirements.txt --no-deps'
Unfortunately this code only support pip installs.  I have no advice for conda users :(

5. Once everything installs, you should be all set to run the following line in terminal:
python unspotable.py eridani.cfg

6. The code will run and produce all the plots that can be found currently in eridani-outputs 
NOTE: the above command will overwrite these!

7. Feel free to explore the code further or modify the cfg file to create plots for different stars, spatial resolutions, parameters, etc.

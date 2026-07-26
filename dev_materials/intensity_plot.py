"""
File to document broken intenisty plots - likely not fixable due to how starry runs
https://github.com/pb-aj/un-spot-able
"""

#general imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import starry2 as starry
from cmcrameri import cm
import faulthandler
faulthandler.enable()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
from lib import utils
from lib import fitclass    as fc
from lib.spotable import spotable as se

# py imports
import create_eigens
import create_emaps
import create_rv

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion


# Whether or not to show print statements in code
dpm = True #change to False to make quiet



def intensity_line(realistic_star, ratio_no_limb, phase = 0, lat = 0, fname=None,
                transparent=False, legend=True, title = None, 
                int_border=True, int_labels=True, ticks=True,
                int_gridlines=False, guideline=False, guideline_color="k", 
                fontsize=16, color="sandybrown", include_map=True,
                cmap = cm.buda, map_gridlines=True, map_labels=False, norm=None,
                colorbar="bottom", colorbar_label=True, marker_color="darkgrey"):
    
    """NEED TO TEST MORE EXTRA FEATURES"""

    theta_face = np.linspace(-90,90,91)
    
    theta_all = np.linspace(-180,180,181)[:-1]

    min_range = (theta_all.shape[0] + 1) // 4
    max_range = (theta_all.shape[0] + 1) // 4 * 3 + 1


    realistic_star.map.amp /= ratio_no_limb
    no_limb_intenisty = realistic_star.map.intensity(lat=lat, lon=theta_all,rv=False,limbdarken=False).eval()
    realistic_star.map.amp *= ratio_no_limb

    new_no_limb_intenisty = np.concatenate((no_limb_intenisty[phase//2:],no_limb_intenisty[:phase//2]))

    udeg = realistic_star.map.u.eval()
    mu = np.cos(theta_face * np.pi/180)

    limb_law = 1 - udeg[1]*(1-mu) - udeg[2] * (1-mu)**2

    if include_map:
        fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False,
                                    sharex=False, sharey=False, figsize=(7, 6))

        new_limb_intensity = new_no_limb_intenisty[min_range:max_range] * limb_law
        
        axes[1,0].plot(theta_all[min_range:max_range], new_limb_intensity, label="$I_{limb}$",alpha=1, color=color)
        axes[1,0].plot(theta_all[min_range:max_range], new_no_limb_intenisty[min_range:max_range], ls="-.",alpha=.5,label="$I_{no \ limb}$",c="k")
        

        if int_labels:
            axes[1,0].set_xlabel("Longitude [deg]", fontsize=fontsize)
            axes[1,0].set_ylabel("Intensity [normalized]", fontsize=fontsize)

        if title:
            plt.title(title,fontsize=fontsize*1.5)

        if not int_border:
             axes[1,0].set_frame_on(False)

        if int_gridlines:
            axes[1,0].grid(color=marker_color,linestyle=":")

        if guideline:
            axes[1,0].axvline(x=0, color=guideline_color, linestyle=':', linewidth=1.5)  

        if not ticks:
            axes[1,0].set_xticklabels([])
            axes[1,0].set_yticklabels([])
            axes[1,0].tick_params(left=False, bottom=False)
        else:
            axes[1,0].set_xticks([-60,-30,0,30,60])
            axes[1,0].set_xticklabels([-60,-30,0,30,60])

            axes[1,0].set_xlim(-90,90)

            min_f = np.min(new_limb_intensity)
            max_f = np.max(new_no_limb_intenisty)
            
            center = (max_f + min_f) / 2

            axes[1,0].tick_params(direction="in")

            amp = max_f - min_f

            buffer = 0.05 

            y_tick_values = [min_f, (center + min_f) / 2, center, (center + max_f) / 2, max_f] 
            axes[1,0].set_yticks(y_tick_values)
            axes[1,0].set_yticklabels([f"{val:.2f}" for val in y_tick_values])
            axes[1,0].set_ylim(min_f - amp*buffer, max_f + amp*buffer)

        if legend:
            axes[1,0].legend(loc="upper right")

        if map_labels:
            axes[0,0].set_title("Flux Projection", fontsize=fontsize)


        plt.tight_layout()

        if colorbar_label and colorbar:
            plt.subplots_adjust(hspace=0.25)
            flux_cbar_label = "Flux [Normalized]"
        else:
             flux_cbar_label = None

        if norm is None:
            realistic_star.map.show(theta=phase,rv=False, ax=axes[0,0], latline=lat,
                                            colorbar_label=flux_cbar_label,
                                            colorbar=colorbar, grid=map_gridlines,
                                            cmap=cmap, colorbar_size="2.5%",
                                            file=fname, dpi = 300, transparent=transparent)
        else:
            realistic_star.map.show(theta=phase,rv=False, ax=axes[0,0], latline=lat,
                                            colorbar_label=flux_cbar_label,
                                            colorbar=colorbar, grid=map_gridlines,
                                            cmap=cmap, colorbar_size="2.5%", norm=norm,
                                            file=fname, dpi = 300, transparent=transparent)
        
        plt.close()
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False,
                                    sharex=False, sharey=False, figsize=(12, 5))
    

        new_limb_intensity = new_no_limb_intenisty[min_range:max_range] * limb_law
        
        plt.plot(theta_all[min_range:max_range], new_limb_intensity, label="$I_{limb}$",alpha=1, color=color)
        plt.plot(theta_all[min_range:max_range], new_no_limb_intenisty[min_range:max_range], ls="-.",alpha=.5,label="$I_{no \ limb}$",c="k")

        if int_labels:
            plt.xlabel("Longitude [deg]", fontsize=fontsize)
            plt.ylabel("Intensity [normalized]", fontsize=fontsize)

        if int_gridlines:
            plt.grid(color=marker_color,linestyle=":")

        if title:
            plt.title(title,fontsize=fontsize*1.5)

        if not int_border:
            plt.gca().set_frame_on(False)

        if not ticks:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(left=False, bottom=False)
        else:
            plt.gca().set_xticks([-60,-30,0,30,60])
            plt.gca().set_xticklabels([-60,-30,0,30,60])

            plt.xlim(-90,90)

            min_f = np.min(new_limb_intensity)
            max_f = np.max(new_no_limb_intenisty)
            
            center = (max_f + min_f) / 2

            plt.tick_params(direction="in")

            amp = max_f - min_f

            buffer = 0.05 

            y_tick_values = [min_f, (center + min_f) / 2, center, (center + max_f) / 2, max_f] 
            plt.yticks(y_tick_values)
            plt.gca().set_yticklabels([f"{val:.2f}" for val in y_tick_values])
            plt.ylim(min_f - amp*buffer, max_f + amp*buffer)

        if legend:
            plt.legend(loc="upper right")


        plt.tight_layout()

        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, dpi = 300, transparent=transparent)
        
        plt.close()

def intensity_animations(realistic_star, ratio_no_limb, lat= 0, fname=None,
                 transparent=False, legend=True, title = None, 
                 int_border=True, int_labels=True, ticks=True, 
                 int_gridlines=False, guideline=False, guideline_color="k", 
                 fontsize=16, color="sandybrown", include_map=True,
                 cmap = cm.buda, map_gridlines=True, map_labels=False, norm=None,
                 colorbar="bottom", colorbar_label=True, marker_color="darkgrey",
                 interval=75, fps=10):
    
    """NEED TO TEST EXTRA FEATURES MORE
    also, starry only compute intensity for i = 90 (star frame), so this is not fully accurate
    But the ratio_to_no_limb factor is the same based only on the udeg coeffs"""
    
    theta_face = np.linspace(-90,90,91)
    
    theta_all = np.linspace(-180,180,181)[:-1]

    min_range = (theta_all.shape[0]) // 4
    max_range = (theta_all.shape[0]) // 4 * 3 + 1

    realistic_star.map.amp /= ratio_no_limb
    no_limb_intenisty = realistic_star.map.intensity(lat=lat, lon=theta_all,rv=False,limbdarken=False).eval()
    realistic_star.map.amp *= ratio_no_limb

    udeg = realistic_star.map.u.eval()
    mu = np.cos(theta_face * np.pi/180)

    limb_law = 1 - udeg[1]*(1-mu) - udeg[2] * (1-mu)**2

    intensity_info = [min_range,max_range,limb_law, theta_all[::2].astype(int)]

    if include_map:

        fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False,
                                    sharex=False, sharey=False, figsize=(7, 6))
        
        new_limb_intensity = no_limb_intenisty[min_range:max_range] * limb_law

        limb_line = axes[1,0].plot(theta_all[min_range:max_range], new_limb_intensity,label="$I_{limb}$",alpha=1, color=color)
        no_limb_line = axes[1,0].plot(theta_all[min_range:max_range], no_limb_intenisty[min_range:max_range], ls="-.",alpha=.5,label="$I_{no \ limb}$",c="k")
        
        if int_labels:
                    axes[1,0].set_xlabel("Longitude [deg]", fontsize=fontsize)
                    axes[1,0].set_ylabel("Intensity [normalized]", fontsize=fontsize)

        if title:
            plt.title(title,fontsize=fontsize*1.5)

        if not int_border:
             axes[1,0].set_frame_on(False)

        if int_gridlines:
            axes[1,0].grid(color=marker_color,linestyle=":")

        if guideline:
            axes[1,0].axvline(x=0, color=guideline_color, linestyle=':', linewidth=1.5) 

        if not ticks:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(left=False, bottom=False)
        else:
            plt.gca().set_xticks([-60,-30,0,30,60])
            plt.gca().set_xticklabels([-60,-30,0,30,60])

            plt.xlim(-90,90)

        if legend:
            axes[1,0].legend(loc="upper right")

        if map_labels:
            axes[0,0].set_title("Flux Projection", fontsize=fontsize)

        plt.tight_layout()

        if colorbar_label and colorbar:
            plt.subplots_adjust(hspace=0.25)
            flux_cbar_label = "Flux [Normalized]"
        else:
             flux_cbar_label = None

        realistic_star.map.show(theta=np.linspace(0,360,theta_all[::2].shape[0]+1)[:-1],rv=False, ax=axes[0,0], latline=lat,
                                        colorbar_label=flux_cbar_label,
                                        colorbar=colorbar, grid=map_gridlines,
                                        cmap=cmap, colorbar_size="2.5%",
                                        file=fname, dpi = 300, transparent=transparent,
                                        interval=interval, fps=fps, intensity_info=intensity_info,
                                        extra_lines = [([],limb_line),(no_limb_intenisty,no_limb_line)],
                                        norm=norm)

    
    
    else:
        def update_intensity(frame, no_limb_intenisty):

            new_no_limb_intenisty = np.concatenate((no_limb_intenisty[frame:],no_limb_intenisty[:frame]))[min_range:max_range]

            new_limb_intensity = new_no_limb_intenisty * limb_law
            
            limb_line[0].set_data(theta_all[min_range:max_range], new_limb_intensity)
            
            
            no_limb_line[0].set_data(theta_all[min_range:max_range], new_no_limb_intenisty)
            
            return limb_line[0], no_limb_line[0]

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        limb_line = ax.plot([], [],label="$I_{limb}$",alpha=1, color=color)
        no_limb_line = ax.plot([], [], ls="-.",alpha=.5,label="$I_{no \ limb}$",c="k")

        ani = animation.FuncAnimation(
                        fig, 
                        update_intensity, 
                        fargs=(no_limb_intenisty,),
                        frames=theta_all[::2].astype(int), 
                        interval=interval, 
                        blit=True
                    )
        
        if int_labels:
            plt.xlabel("Longitude [deg]", fontsize=fontsize)
            plt.ylabel("Intensity [normalized]", fontsize=fontsize)

        if title:
            plt.title(title,fontsize=fontsize*1.5)

        if int_gridlines:
            plt.grid(color=marker_color,linestyle=":")

        if not int_border:
            plt.gca().set_frame_on(False)

        if not ticks:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(left=False, bottom=False)
        else:
            plt.gca().set_xticks([-60,-30,0,30,60])
            plt.gca().set_xticklabels([-60,-30,0,30,60])

            plt.xlim(-90,90)

            plt.ylim(0,1)

        if legend:
            plt.legend(loc="upper right")

        plt.tight_layout()

        if fname is None:
            HTML(ani.to_html5_video())
        else:
            if fname.endswith("gif"):
                if transparent:
                    ani.save(fname, writer="imagemagick", dpi=300, fps=fps, 
                            savefig_kwargs={"transparent": True})
                else:
                    ani.save(fname, writer="pillow", dpi=300, fps=fps, savefig_kwargs={"transparent": False})
            else:
                ani.save(fname, writer="ffmpeg", dpi=300)
        
    plt.close()



# intensity_line(realistic_star, ratio_no_limb, phase = phase, lat = lat, cmap=cmap,
#             fname=f"{results_path}/{folder_name}/emap_intensity_l{lat}_p{phase}.png")

# intensity_animations(realistic_star, ratio_no_limb, lat=lat, cmap=cmap,
#             fname=f"{results_path}/{folder_name}/emap_intensity_animation_l{lat}.gif")

# intensity_animations(realistic_star, ratio_no_limb, lat=lat, cmap=cmap,
#             fname=f"{results_path}/{folder_name}/emap_intensity_animation_l{lat}.mp4")



def multi_phase_plot(rv_star,fname=None,cmap=cm.bam,norm=None):
    """
    Function to generate plot of rv_star at different phase to show both flux value and rv value
    **NOTE** This function is not super developed as it was useful in code development but not needed in final versions.

    Arguments
    ---------
    rv_star: object
        A starry star object, initialized with a cfg file

    fname: string (optional)
        Name to save figure as, but if None will display figure.  Default is None

    cmap: str (optional)
        What color map to use in plots.  Default is cm.bam
        To use the original starry colors, set to 'plasma'
    
    norm: Matplotlib Normalization (optional)
        Normalization to use for map, if None uses no norm.  Default is None

    Returns
    -------
    None
    """

    nrows = 3
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols*2, squeeze=False,
                                sharex=False, sharey=False, figsize=(12, 6))
    
    degree = 0
    number_plots = nrows*ncols
    for j in range(number_plots):

        xloc = j %  ncols
        yloc = j // ncols
        ax = axes[yloc, xloc]
        ax2 = axes[yloc, xloc+ncols]

        rv_star.map.show(rv=False,theta=degree,ax=ax,figsize=(5,5),cmap=cmap,norm=norm)
        ax.set_title(f"{rv_star.map.flux(theta=degree).eval()[0]:.3g}")

        rv_star.map.show(rv=True,theta=degree,ax=ax2,figsize=(5,5),cmap=cm.vik)
        ax2.set_title(f"{rv_star.map.rv(theta=degree).eval()[0]:.3g}")

        degree += int(360/number_plots)
    
    fig.text(.22, 0.95, f"Flux Maps",fontsize="large")
    fig.text(.72, 0.95, f"RV Maps",fontsize="large")
    fig.suptitle(f"Plots range from 0 to 360 in {int(360/number_plots)} degree intervals.",y=0.02)
    fig.tight_layout()
    if fname:
        plt.savefig(fname,bbox_inches="tight",dpi=300)
    else:
        plt.show()
    plt.close(fig)

import matplotlib as mlib
import matplotlib.pyplot as plt
import numpy as np

from core import *

# code edited from https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values

def render(objects, title=None, save_path=None, show=True):
    """Plot 2d matrix with grid with well-defined colors for specific boundary values.

    :param data: 2d matrix
    :param title: title
    :param save_path: save path
    :param show: whether plot should be plotted
    """
    colors=['white', 'beige', 'yellow', 'orange', 'blue', 'brown', 'green', 'purple', 'black', 'pink'] # until 'blue', it's for fire
    colors+=['sandybrown', 'salmon', 'silver', 'cadetblue', 'aqua', 'saddlebrown', 'magenta']
    color=lambda c : colors.index(c)+1e-1

    bounds=list(range(len(colors)+1))

    assert len(bounds)-1==len(colors), f"Number of bounds {len(bounds)} should be one greater then amt of colors {len(colors)}"

    # create data matrix from 
    # plot data matrix
    data=np.zeros((Coordinate.HEIGHT, Coordinate.WIDTH))
    for ob in objects:
        # NOTE: z axis here
        xx,yy,zz=ob.position.get()
        if isinstance(ob, Flammable):
            # ranging from beige to blue in intensity
            data[yy][xx]=(ob.intensity.value+1e-1) # Enum is 1-indexed

        elif isinstance(ob, AbsAgent):
            # pink agent
            data[yy][xx]=color('pink')

        elif isinstance(ob, Reservoir):
            # change color depending on type
            if ob.type=='A':
                # darker blue for sand
                data[yy][xx]=color('cadetblue')
            elif ob.type=='B':
                # obnoxious blue for water 
                data[yy][xx]=color('aqua')

        elif isinstance(ob, Deposit):
            # always black
            data[yy][xx]=color('black')

        elif isinstance(ob, Person):
            # purple (because holding breath so much)
            if ob.grabbed:
                # if being carried magenta (to indicate that absagent - pink - is carrying it)
                data[yy][xx]=color('magenta')
            elif not ob.grabbed and not ob.deposited:
                data[yy][xx]=color('purple')
            else:
                pass

        elif isinstance(ob, Fire):
            if ob.fire_type=='A':
                # sandy for requiring sand
                data[yy][xx]=color('saddlebrown')
            elif ob.fire_type=='B':
                # salmon for requiring water
                data[yy][xx]=color('salmon')
        else:
            data[yy][xx]=color('white')

    # create discrete colormap
    cmap = mlib.colors.ListedColormap(colors)
    norm = mlib.colors.BoundaryNorm(bounds, cmap.N)

    # enable or disable frame
    plt.figure(frameon=True)

    if title is not None: plt.title(title)

    # disable labels
    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # show grid
    plt.grid(axis='both', color='k', linewidth=2) 
    plt.xticks(np.arange(0.5, data.shape[1], 1))  # correct grid sizes
    plt.yticks(np.arange(0.5, data.shape[0], 1))

    plt.imshow(data, cmap=cmap, norm=norm)

    if show:
        # display main axis 
        plt.show()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()

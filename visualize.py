from PIL import Image
import inspect

def display_frames_as_gif(frames):
    stack = inspect.stack()
    p = inspect.getmodule(stack[1][0]).__file__

    filename = p[p.rfind('/')+1:p.rfind('.')]

    di = p[:p.rfind('/')+1]

    frs = [Image.fromarray(f, mode='RGB') for f in frames]
    frs[0].save(di+filename+'.gif', save_all=True, append_images=frs[1:], optimize=False, duration=40, loop=0)
# From https://stackoverflow.com/questions/40132630/python-cprofiler-a-function
# CC-BY-SA
import cProfile
import pstats
from io import StringIO


def qprofile(func):
    def profiled_func(*args, **kwargs):
        if 'profile' in kwargs and kwargs['profile']:
            kwargs.pop('profile')
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                s = StringIO()
                ps = pstats.Stats(profile, stream=s).strip_dirs().sort_stats('cumulative')
                ps.print_stats(30)
                print(s.getvalue())
        else:
            result = func(*args, **kwargs)
            return result
    return profiled_func

#@qprofile
#def process_one(cmd):
#    import commands
#    output = commands.getoutput(cmd)
#    return output

# Function is profiled if profile=True in kwargs
#print(process_one('uname -a', profile=True))

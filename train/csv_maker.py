# -*- coding:utf-8 -*-
import os
import gzip


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


### Audio tagging (AT) related io. 
def at_write_prob_mat_to_csv(na_list, prob_mat, out_path):
    """Write out audio tagging (AT) prediction probability to a csv file. 
    
    Args:
      na_list: list of names. 
      prob_mat: ndarray, (n_clips, n_labels). 
      out_path: string, path to write out the csv file. 
      
    Returns:
      None
    """
    create_folder(os.path.dirname(out_path))
    f = gzip.open(out_path, 'w')
    for n in range(len(na_list)):
        na = na_list[n]
        f.write(na)
        for p in prob_mat[n]:
            f.write('\t' + "%.3f" % p)
        f.write('\r\n')
    f.close()

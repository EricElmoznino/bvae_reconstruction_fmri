#! /bin/sh
cd $1
git clone https://github.com/deepmind/dsprites-dataset.git dsprites
cd dsprites
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
mv *.npz data.npz

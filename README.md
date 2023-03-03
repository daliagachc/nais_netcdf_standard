# install 


```bash
cd /tmp

#rm -r /tmp/nais_netcdf_standard
git clone https://github.com/daliagachc/nais_netcdf_standard.git

cd nais_netcdf_standard

conda create -y --name nais_necdf python
conda install --force-reinstall -y -q --name nais_necdf -c conda-forge --file requirements.txt
conda activate nais_necdf
conda develop install src

#test 
python -i ./src/nais_netcdf/nbs/z01_test.py

#conda deactivate; conda env remove --name nais_necdf;rm -rf /tmp/nais_netcdf_standard

```

# test

the main functions are used in the notebooks inside [nbs](src%2Fnais_netcdf%2Fnbs)


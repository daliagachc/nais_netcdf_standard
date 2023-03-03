# install 

- download 

- conda install

```sh
cd /tmp

#git clone https://github.com/daliagachc/nais_netcdf_standard.git

cd nais_netcdf_standard

conda create -y --name nais_necdf python
conda install --force-reinstall -y -q --name nais_necdf -c conda-forge --file requirements.txt
conda activate nais_necdf
conda develop install src

#donda deactivate 
#conda env remove --name nais_necdf
```


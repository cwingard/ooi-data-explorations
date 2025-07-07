# ooi-data-explorations
Explorations of Ocean Observatories Initiative Datasets via MATLAB, Python, R, and Julia.

### matlab
This subfolder contains a downloadable matlab toolbox that consists of three primary functions for requesting and accessing OOI data.
Installation instructions can be found in the README in that folder.

### python
This subfolder contains a downloadable python package that consists of functions that can request, download, and clean up OOI datasets.
Installation instructions can be found in the README in that folder.

### R
This subfolder contains a series of examples that utilize the ooim2mr package.
Links to the R package and installation instructions can be found in the README in that folder.

### Julia
This subfolder contains a working set of functions that allows Julia users to download OOI data via the OOI M2M interface.


### OOI Nomenclature

### OOI Sites
A comprehensive list of OOI assets can be found [here](https://oceanobservatories.org/site-list/).

### Instrument Classes
[**ADCPT**](https://oceanobservatories.org/instrument-series/adcpta/) - Teledyne RDI - WorkHorse  
[**CTDBP**](https://oceanobservatories.org/instrument-series/ctdbpc/) - SBE - 16plusV2  
[**DOSTA**](https://oceanobservatories.org/instrument-series/dostad/) - Aanderaa - Optode 4831  
[**FLORT**](https://oceanobservatories.org/instrument-series/flortd/) - WET Labs - ECO Triplet-w  
[**METBK**](https://oceanobservatories.org/instrument-series/metbka/) - Star Engineering - ASIMET  
[**NUTNR**](https://oceanobservatories.org/instrument-series/nutnrb/) - SBE - SUNA V2  
[**OPTAA**](https://oceanobservatories.org/instrument-series/optaad/) - SBE - AC-S  
[**PCO2A**](https://oceanobservatories.org/instrument-series/pco2aa/) - Pro-Oceanus - pCO2-pro  
[**PCO2W**](https://oceanobservatories.org/instrument-series/pco2wb/) - Sunburst - SAMI-pCO2  
[**PHSEN**](https://oceanobservatories.org/instrument-series/phsend/) - Sunburst - SAMI-pH  
[**SPKIR**](https://oceanobservatories.org/instrument-series/spkirb/) - SBE - OCR507  
[**VELPT**](https://oceanobservatories.org/instrument-series/velpta/) - Nortek - Aquadopp  
[**WAVSS**](https://oceanobservatories.org/instrument-series/wavssa/) - Axys Technologies - TRIAXYS


## Contributing

Users are encouraged to contribute to this code. The hope is this repository can provide the science community with a 
means of accessing and working with the OOI data. To contribute, please fork the main 
[repo](https://github.com/oceanobservatories/ooi-data-explorations) to your own GitHub account, create a branch, do 
your work, and then (when satisfied) submit a pull request to have your work integrated back into the main project repo.

```bash
# Git workflow template for working with the OOI Data Explorations repository.

# Create your development directories (just a guide, use your own directories)
mkdir -p ~/Documents/GitHub
cd ~/Documents/GitHub

# Fork the oceanobservatories/ooi-data-explorations repository to your account 
# and clone a copy of your fork to your development machine.
git clone git@github.com:<your_account>/ooi-data-explorations.git
 
# The next steps must be completed in the local repository directory
cd ooi-data-explorations
 
# Add the upstream feed for the master repository
git remote add upstream git@github.com:oceanobservatories/ooi-data-explorations.git
git fetch upstream
```
>***Did you encounter an error when you trying to fetch upstream?*** *GitHub deprecated password-based authentication for Git operations (as of August 2021), encouraging users to adopt more secure methods like PATs and SSH keys. To fix this error, try adding a new SSH key to your github account. Once you have set up a SSH key, try the `git fetch upstream` command again. Additional details can be found [here.](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)*

```
# Set the local master to point instead to the upstream master branch
git branch master --set-upstream-to upstream/master

# Keep your master branch updated, tied to the upstream master, and
# keep your remote fork in sync with the official repository (do this
# regularly)
git pull --ff-only upstream master
git push origin master

# Create your feature branch based off of the most recent version of the master
# branch by starting a new branch via...
#    git checkout master
#    git pull
#    git push origin master
# ... and then:
git checkout -b <branch>

### --- All of the next steps assume you are working in your <branch> --- ###
# Do your work, making incremental commits as/if needed, and back up to your
# GitHub repository as/if needed.
while working == true
    git add <files>
    git commit -am "Commit Message"
    git push origin <branch>
end

# Before pushing your final changes to your repository, rebase your changes
# onto the latest code available from the upstream master.
git fetch upstream
git rebase -p upstream/master

# At this point you will need to deal with any conflicts, of which there should
# be none. Hopefully...

# Push the current working, rebased branch to your GitHub fork and then 
# make a pull request to merge your work into the main code branch. Once the
# pull request is generated, add a comment with the following text:
#
#    @<code_admin> Ready for review and merge
#
# This will alert the main code admin to process the pull request.
git push -f origin <branch>
 
# At this point you can switch back to your master branch. Once the pull
# request has been merged into the main code repository, you can delete
# your working branches both on your local machine and from your GitHub
# repository.
git checkout master
git pull
git push origin master
git branch -D <branch>
git branch -D origin/<branch>
```

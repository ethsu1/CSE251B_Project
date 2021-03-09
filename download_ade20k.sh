mkdir -p real_data;
wget -O ./real_data/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip;
unzip ./real_data/ADEChallengeData2016.zip -d ./real_data;
rm ./real_data/ADEChallengeData2016.zip;
mv ./real_data/ADEChallengeData2016/images/training/* real_data;
rm -r ./real_data/ADEChallengeData2016;
echo "Dataset downloaded."
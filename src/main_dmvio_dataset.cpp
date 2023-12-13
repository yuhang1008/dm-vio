/**
* This file is based on the file main_dso_pangolin.cpp of the project DSO written by Jakob Engel.
* It has been modified by Lukas von Stumberg for the inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
*
* Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

// Main file for running on datasets, based on the main file of DSO.

#include "util/MainSettings.h"
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "dso/util/settings.h"
#include "dso/util/globalFuncs.h"
#include "dso/util/DatasetReader.h"
#include "dso/util/globalCalib.h"
#include "util/TimeMeasurement.h"

#include "dso/util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include <util/SettingsUtil.h>

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

std::string gtFile = "";
std::string source = "";
std::string imuFile = "";

bool reverse = false;
int start = 0;
int end = 100000;
int maxPreloadImages = 0; // If set we only preload if there are less images to be loade.
bool useSampleOutput = false;

using namespace dso;


dmvio::MainSettings mainSettings;
dmvio::IMUCalibration imuCalibration;
dmvio::IMUSettings imuSettings;


// from ChatGPT: Together, these 2 functions provide a way to gracefully handle the termination of a program in response to
// a SIGINT signal, allowing for custom cleanup actions (such as printing a message) before exiting with an error status.

// SIGINT is a signal in Unix-like operating systems, including Linux and macOS, that stands for "Signal Interrupt."

// so the only function of the code provided is to show ctrl+c caught when the user press ctrl+c?

// ChatGPT:
// Yes, the main purpose of the code you provided is to catch and handle the Ctrl+C signal (SIGINT) when the user presses Ctrl+C in the terminal.

void my_exit_handler(int s)
{
    printf("Caught signal %d\n", s);
    exit(1);
}

void exitThread()
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    while(true) pause();
}


void run(ImageFolderReader* reader, IOWrap::PangolinDSOViewer* viewer)
{

    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }


    int lstart = start; // start time?
    int lend = end; // end time?
    int linc = 1; // frame difference
    if(reverse) // reverse to play the data (so strange)
    {
        assert(!setting_useIMU); // Reverse is not supported with IMU data at the moment!
        printf("REVERSE!!!!");
        lstart = end - 1;
        if(lstart >= reader->getNumImages())
            lstart = reader->getNumImages() - 1;
        lend = start;
        linc = -1;
    }

    // playbackSpeed=0 for linearize (play as fast as possible, while sequentializing tracking & mapping). 
    // otherwise, factor on timestamps.
    bool linearizeOperation = (mainSettings.playbackSpeed == 0);

    // if non realtime mode, can set minimum frames between key frames?
    if(linearizeOperation && setting_minFramesBetweenKeyframes < 0) 
    {   
        //setting_minFramesBetweenKeyframes default is -0.5, set to 0.5?
        setting_minFramesBetweenKeyframes = -setting_minFramesBetweenKeyframes;
        std::cout << "Using setting_minFramesBetweenKeyframes=" << setting_minFramesBetweenKeyframes
                  << " because of non-realtime mode." << std::endl;
    }

    // note, imucalibration is initilized already, and all related paras been set into settingsUtil
    // imuSettings has many paras and also been set into settingsUtil
    // above infos set to FullSystem
    FullSystem* fullSystem = new FullSystem(linearizeOperation, imuCalibration, imuSettings);
    fullSystem->setGammaFunction(reader->getPhotometricGamma()); //gamma funcition created from reader

    // visualization avaliable
    if(viewer != 0)
    {   
        //why push back?
        // we found PangolinDSOViewer(class of viewer) and SampleOutputWrapper (calss of sampleOutPutWrapper)
        // are all inheritet from Output3DWrapper
        // probably, viewer is to display all related information
        // while SampleOutputWrapper for 3d point cloud only
        fullSystem->outputWrapper.push_back(viewer);
    }

    // warpper for print 3-d related informations
    std::unique_ptr<IOWrap::SampleOutputWrapper> sampleOutPutWrapper;
    if(useSampleOutput) // default is false
    {
        sampleOutPutWrapper.reset(new IOWrap::SampleOutputWrapper());
        fullSystem->outputWrapper.push_back(sampleOutPutWrapper.get());
    }

    // so the system has one viewer and one sampleOutPutWrapper in the outputWrapper vector ?

    // time for each frame to play at (start from zero), plackbackSpeed is considered
    std::vector<int> idsToPlay;
    std::vector<double> timesToPlayAt; // saved in this vector
    for(int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc)
    {
        idsToPlay.push_back(i);
        if(timesToPlayAt.size() == 0) 
        {
            timesToPlayAt.push_back((double) 0); // start from 0
        }else
        {
            double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
            double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
            timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / mainSettings.playbackSpeed); //playbackSpeed=1 for prest is not 0
        }
    }

    if(mainSettings.preload && maxPreloadImages > 0)
    {
        if(reader->getNumImages() > maxPreloadImages)
        {
            printf("maxPreloadImages EXCEEDED! NOT PRELOADING!\n");
            mainSettings.preload = false;
        }
    }

    // preload images
    std::vector<ImageAndExposure*> preloadedImages;
    if(mainSettings.preload)
    {
        printf("LOADING ALL IMAGES!\n");
        for(int ii = 0; ii < (int) idsToPlay.size(); ii++)
        {
            int i = idsToPlay[ii];
            preloadedImages.push_back(reader->getImage(i));
        }
    }

    struct timeval tv_start; 
    gettimeofday(&tv_start, NULL);
    clock_t started = clock();
    double sInitializerOffset = 0;

    // load ground truth data
    bool gtDataThere = reader->loadGTData(gtFile);

    bool imuDataSkipped = false;
    dmvio::IMUData skippedIMUData;// skiped imu data because the frame is skiped


    // main loop for dm-vio
    for(int ii = 0; ii < (int) idsToPlay.size(); ii++)
    {
        if(!fullSystem->initialized)    // if not initialized: reset start time.
        {
            gettimeofday(&tv_start, NULL);
            started = clock();
            sInitializerOffset = timesToPlayAt[ii]; // update sInitializerOffset util useful timesToPlayAt (wrt starting time (0))
        }

        int i = idsToPlay[ii]; // current image id

        // read current image
        ImageAndExposure* img;
        if(mainSettings.preload)
            img = preloadedImages[ii];
        else
            img = reader->getImage(i);

        bool skipFrame = false;

        // this if make sure data play as original time if mainSettings.playbackSpeed != 0
        if(mainSettings.playbackSpeed != 0) //playbackSpeed=1 for prest is not 0, which will not be real-time
        {
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);

            // sSinceStart is time wrt to first initialized frame
            double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) +
                                                       (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

            //sleep for a while if data read too quick
            if(sSinceStart < timesToPlayAt[ii])
                usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
                                                // plus a stange threshold
            else if(sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
            {
                printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                skipFrame = true;
            }
        }

        // read gt data
        dmvio::GTData data;
        bool found = false;
        if(gtDataThere)
        {
            data = reader->getGTData(i, found);
        }

        // read imu data associated to the frame
        std::unique_ptr<dmvio::IMUData> imuData;
        //only read imu data when useIMU is set true
        if(setting_useIMU)
        {
            imuData = std::make_unique<dmvio::IMUData>(reader->getIMUData(i));
        }
        
        if(!skipFrame)
        {   
        
            if(imuDataSkipped && imuData)
            {   
                // insert IMU data that is skipped with the skipped frame
                imuData->insert(imuData->begin(), skippedIMUData.begin(), skippedIMUData.end());
                skippedIMUData.clear();
                imuDataSkipped = false;
            }
            // addActiveFrame is a very important function 
            // variables: image, id, IMUData from last until this image?, gtdata(optional)
            fullSystem->addActiveFrame(img, i, imuData.get(), (gtDataThere && found) ? &data : 0);
            if(gtDataThere && found && !disableAllDisplay)
            {
                viewer->addGTCamPose(data.pose);
            }
        }else if(imuData)
        {
            imuDataSkipped = true;
            skippedIMUData.insert(skippedIMUData.end(), imuData->begin(), imuData->end());
        }


        delete img;

        if(fullSystem->initFailed || setting_fullResetRequested)
        {   //?
            if(ii < 250 || setting_fullResetRequested)
            {
                printf("RESETTING!\n");
                std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                delete fullSystem;
                for(IOWrap::Output3DWrapper* ow : wraps) ow->reset(); //?

                fullSystem = new FullSystem(linearizeOperation, imuCalibration, imuSettings);
                fullSystem->setGammaFunction(reader->getPhotometricGamma());
                fullSystem->outputWrapper = wraps;

                setting_fullResetRequested = false;
            }
        }


        if(viewer != nullptr && viewer->shouldQuit())
        {
            std::cout << "User closed window -> Quit!" << std::endl;
            break;
        }

        if(fullSystem->isLost)
        {
            printf("LOST!!\n");
            break;
        }

    } // end of main loop

    fullSystem->blockUntilMappingIsFinished();
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);


    fullSystem->printResult(imuSettings.resultsPrefix + "result.txt", false, false, true);
    fullSystem->printResult(imuSettings.resultsPrefix + "resultKFs.txt", true, false, false);
    fullSystem->printResult(imuSettings.resultsPrefix + "resultScaled.txt", false, true, true);

    dmvio::TimeMeasurement::saveResults(imuSettings.resultsPrefix + "timings.txt");


    int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
    double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
    double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC);
    double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                       (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
    printf("\n======================"
           "\n%d Frames (%.1f fps)"
           "\n%.2fms per frame (single core); "
           "\n%.2fms per frame (multi core); "
           "\n%.3fx (single core); "
           "\n%.3fx (multi core); "
           "\n======================\n\n",
           numFramesProcessed, numFramesProcessed / numSecondsProcessed,
           MilliSecondsTakenSingle / numFramesProcessed,
           MilliSecondsTakenMT / (float) numFramesProcessed,
           1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
           1000 / (MilliSecondsTakenMT / numSecondsProcessed));
    fullSystem->printFrameLifetimes();
    if(setting_logStuff)
    {
        std::ofstream tmlog;
        tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
        tmlog << 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC * reader->getNumImages()) << " "
              << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                 (float) reader->getNumImages() << "\n";
        tmlog.flush();
        tmlog.close();
    }

    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
    }


    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");
}


int main(int argc, char** argv)
{
    setlocale(LC_ALL, "C");

#ifdef DEBUG
    std::cout << "DEBUG MODE!" << std::endl;
#endif

    bool use16Bit = false;

    auto settingsUtil = std::make_shared<dmvio::SettingsUtil>();

    // register default values to settingsUtil
    imuSettings.registerArgs(*settingsUtil); //this is actually the system setting, not imu setting
    imuCalibration.registerArgs(*settingsUtil); // IMU calibration setting
    mainSettings.registerArgs(*settingsUtil);

    // Dataset specific arguments. For other commandline arguments check out MainSettings::parseArgument,
    // MainSettings::registerArgs, IMUSettings.h and IMUInitSettings.h
    settingsUtil->registerArg("files", source);
    settingsUtil->registerArg("start", start);
    settingsUtil->registerArg("end", end);
    settingsUtil->registerArg("imuFile", imuFile);
    settingsUtil->registerArg("gtFile", gtFile);
    settingsUtil->registerArg("sampleoutput", useSampleOutput);
    settingsUtil->registerArg("reverse", reverse);
    settingsUtil->registerArg("use16Bit", use16Bit);
    settingsUtil->registerArg("maxPreloadImages", maxPreloadImages);

    // This call will parse all commandline arguments and potentially also read a settings yaml file if passed.
    mainSettings.parseArguments(argc, argv, *settingsUtil);

    if(mainSettings.imuCalibFile != "")
    {
        imuCalibration.loadFromFile(mainSettings.imuCalibFile);
    }

    // Print settings to commandline and file.
    std::cout << "Settings:\n";
    settingsUtil->printAllSettings(std::cout);
    {
        std::ofstream settingsStream;
        settingsStream.open(imuSettings.resultsPrefix + "usedSettingsdso.txt");
        settingsUtil->printAllSettings(settingsStream);
    }

    // hook crtl+C.
    // This line creates a new Boost.Thread object named exThread and initializes it with a thread that runs the exitThread function. 
    // the only function of the code provided is to show ctrl+c caught when the user press ctrl+c 
    boost::thread exThread = boost::thread(exitThread);

    // read image from zip files and push the names to variable 'files'
    // load timestamps and exposure times, as well as ids
    //                                             image_folder calibration_file   gama_calib_file            vignette_file         ??
    ImageFolderReader* reader = new ImageFolderReader(source, mainSettings.calib, mainSettings.gammaCalib, mainSettings.vignette, use16Bit);
    // load IMU data 
    reader->loadIMUData(imuFile);
    // set camera parameters as well as pyramid parameters!
    reader->setGlobalCalibration();


    if(!disableAllDisplay)
    {   
        // inita viewer                                              wG[0] = w;  hG[0] = h;
        IOWrap::PangolinDSOViewer* viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false, settingsUtil,
                                                                          nullptr);

        // boost::thread constructor expects a callable object with no arguments, so use bind to pass the arguments
        boost::thread runThread = boost::thread(boost::bind(run, reader, viewer));

        viewer->run();

        delete viewer;

        // This call makes sure the main thread waits for the runThread to finish before moving on. 
        runThread.join();
    }else
    {
        run(reader, 0);
    }


    return 0;
}

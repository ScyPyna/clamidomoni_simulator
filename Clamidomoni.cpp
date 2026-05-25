//simulation of active Janus particles in a speckle field - 2022 Davide Breoni

#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <valarray>
#include <atomic>
#include <time.h>
#include "Clamidomoni.h"
#include <random>
#include <string>
#include <cstring>
#include <fstream>
#define RYML_SINGLE_HDR_DEFINE_NOW
#include "rapidyaml.h"

using namespace std;

#define PI 3.1415926


void lama(int i) //just for debugging
{
  std::cout << "lama " << i << endl;
}


void multi_particle_simulation(simulation &sim) //function that contains the simulation
{
  sim.init_sim();



  for (int i=1; i<sim.par.Nsim+1; i++)
  {
    //cout << "step=" << i << endl;
    sim.md_step_euler(i);
  }
  //sim.MSD();
  sim.MSD_timeavg();
  sim.MD_timeavg();

}

averaging_data make_averaging_data(parameters par, string name) //creates observables that are automatically averaged during the simulation
{
  averaging_data ret;
  ret.par=par;
  ret.arraytot=valarray<double> (0.0,par.Nlevel+1);
  ret.arrayvartot=valarray<double> (0.0,par.Nlevel+1);
  ret.oldarray=valarray<double> (0.0,par.Nlevel+1);
  ret.name=name;

  return ret;
}

parameters read_parameters() //reads parameters
{
  std::ifstream config_file("parameters_multi.yaml");
  std::string config_string{
    std::istreambuf_iterator<char>(config_file),
    std::istreambuf_iterator<char>()
  };
  ryml::Tree config_tree = ryml::parse_in_place(ryml::to_substr(config_string));
  config_tree.resolve();

  auto config = config_tree.rootref();

  parameters ret;

  config["part_diam"] >> ret.part_diam;
  config["active_speed"] >> ret.active_speed;
  config["kBT"] >> ret.kBT;
  config["eta"] >> ret.eta;
  config["sub_dist"] >> ret.sub_dist;
  config["dt"] >> ret.dt;
  config["NperCore"] >> ret.NperCore;
  config["power_input"] >> ret.power_input;
  config["fk"] >> ret.fk;
  config["HOMOGENEOUS"] >> ret.HOMOGENEOUS;
  config["GAUSSIAN"] >> ret.GAUSSIAN;
  config["LINEAR"] >> ret.LINEAR;
  config["LARGE"] >> ret.LARGE;
  config["SYMMETRIZED"] >> ret.SYMMETRIZED;
  config["EXPERIMENT"] >> ret.EXPERIMENT;
  config["SAME_PEAK"] >> ret.SAME_PEAK;
  config["PBC"] >> ret.PBC;
  config["PUNCTIFORM"] >> ret.PUNCTIFORM;
  config["WCA"] >> ret.WCA;
  config["NOCENTER"] >> ret.NOCENTER;
  config["Nlevel"] >> ret.Nlevel;
  config["Nblock"] >> ret.Nblock;
  config["rho"] >> ret.rho;
  config["LJepsilon"] >> ret.LJepsilon;
  config["avg_tbr"] >> ret.avg_tbr;
  config["INDEPENDENT"] >> ret.INDEPENDENT;
  config["COLLISIONS"] >> ret.COLLISIONS;
  config["TBR_I"] >> ret.TBR_I;
  config["TBR_THETA"] >> ret.TBR_THETA;
  config["TBR_DIST"] >> ret.TBR_DIST;
  config["DELAY"] >> ret.DELAY;
  config["coll_tbr_rate"] >> ret.coll_tbr_rate;
  config["grad_amplification"] >> ret.grad_amplification;
  config["grad_sign_thresh"] >> ret.grad_sign_thresh;
  config["threshold_std"] >> ret.threshold_std;
  config["threshold_decay_rate"] >> ret.threshold_decay_rate;
  config["grad_decay_rate"] >> ret.grad_decay_rate;
  config["grad_sign_dist"] >> ret.grad_sign_dist;
  config["grad_dependence"] >> ret.grad_dependence;
  config["tbr_amplification"] >> ret.tbr_amplification;
  config["total_tbr_multiplier"] >> ret.total_tbr_multiplier;
  config["v_reduction"] >> ret.v_reduction;
  config["upper_mem_thresh"] >> ret.upper_mem_thresh;
  config["lower_mem_thresh"] >> ret.lower_mem_thresh;
  config["time_mem"] >> ret.time_mem;
  config["rec_jump"] >> ret.rec_jump;
  config["S_model"] >> ret.S_model;
  config["S_tau"] >> ret.S_tau;
  config["S_gamma"] >> ret.S_gamma;
  config["S_st"] >> ret.S_st;
  config["S_stot"] >> ret.S_stot;
  config["ABP_model"] >> ret.ABP_model;
  config["ABP_torque_fac"] >> ret.ABP_torque_fac;

  ret.update_parameters();

  return ret;

}

simulation make_simulation(parameters par, tumbles tbr_all, tumbles ill_all, int sim_number) //function that creates the simulation object
{
  simulation ret;
  mt19937_64 mt;
  mt.seed(sim_number);
  normal_distribution<double> norm_distribution(0.0,1.0);
  uniform_real_distribution<double> uni_distribution(0.0,1.0);
  //cout << uni_distribution(mt) <<endl;
  ret.mt= mt;
  ret.norm_distribution=norm_distribution;
  ret.uni_distribution=uni_distribution;

  //par.part_diam=par.part_diam*(1+sim_number);//here we add modificator to test more stuff for each simulation
  //par.update_parameters();

  ret.par = par;
  ret.tbr_all = tbr_all;
  ret.ill_all = ill_all;
  ret.sim_number= sim_number;

  ret.M1=ret.make_moment();
  ret.M2=ret.make_moment();
  return ret;
}

int main(int argc, char* argv[]) //main
{

    int N, i, j, k, sim_number, level;

    parameters par;
    
    par=read_parameters();
    
    //kBT=1.380649*3e-9;//thermal energy at 300K in kg*mum^2/s^2
    //eta=1e-9;//water viscosity in kg/(mum*s)
//
    //part_diam=argc>1?atof(argv[1]):3.25; //in micrometri
    //D=argc>2?atof(argv[2]):kBT/(3*PI*part_diam*eta);
    //D_r=argc>3?atof(argv[3]):3*D/(part_diam*part_diam); 
    //Ir=argc>4?atof(argv[4]):0.5848585690515807;
    //dt=argc>5?atof(argv[5]):0.01;
    //NperCore=argc>6?atoi(argv[6]):7;
    //c1=argc>7?atof(argv[7]):0.6*pow(part_diam,2)/D;//0.6 from Bechinger
    //c2=argc>8?atof(argv[8]):-1.2*pow(part_diam,2)/D;
    //QUENCH=argc>9?atoi(argv[9]):1; //0 for all different maps, 1 for only one map
    //HOMOGENEOUS=argc>10?atoi(argv[10]):0; //1 for homogeneous motility
    //
    //Nlevel=18; 
    //Nblock=2;
    //Nsim=pow(Nblock,Nlevel);

    const auto processor_count= 1;
    //const auto processor_count= thread::hardware_concurrency();

    averaging_data M1, M2, E;

    M1=make_averaging_data(par, "M1");
    M2=make_averaging_data(par, "M2");

    N=processor_count*par.NperCore;

    vector<simulation> simvec (processor_count);
    vector<thread> threads (processor_count);
    
    tumbles tbr_all, ill_all;

    tbr_all.sim_number=0;
    tbr_all.par=par;
    tbr_all.read_field(0);

    ill_all.sim_number=0;
    ill_all.par=par;
    ill_all.read_field(1);

    for (j=0 ; j<par.NperCore; j++) //assigns a simulation to each thread
    {
      for (i=0; i<processor_count; i++)
      {
        sim_number=i+j*processor_count;
        simvec[i]=make_simulation(par, tbr_all, ill_all, sim_number);
        threads[i]=thread(multi_particle_simulation,std::ref(simvec[i])); //normally references could be simply called with "sum", in a thread we also need "std::ref"
      }


      if (j%1==0)
        printf("%d\n", j);

      for (i=0; i<processor_count; i++) //threads are joined
      {
        threads[i].join();
      }
      for (i=0; i<processor_count; i++)
      {
        sim_number=i+j*processor_count;

        if (sim_number<N) //prints logs
        {
          simvec[i].print_log();
          simvec[i].print_quantities();
        }

        M1.incremental_average(simvec[i], simvec[i].M1);
        M2.incremental_average(simvec[i], simvec[i].M2);
  
      }
      
    }

    
    M1.normalize_variance(N);
    M2.normalize_variance(N);


    M1.print_array();
    M2.print_array();


    return 0;
}

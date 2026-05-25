#include <valarray>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include <complex>

#define PI  3.1415926

using namespace std;

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct parameters{
  double dt;
  double D;
  double D_r;
  double Ir;
  double part_diam;
  double active_speed;
  double kBT;
  double eta;
  double sub_dist;
  double gamma;
  double gamma1;
  double beta;
  double beta1;
  double rho;
  double LJcutoff;
  double LJepsilon;
  double max_force;
  double LJoffset;
  double avg_tbr;
  double sub_ratio;
  double Bt;
  double Br;
  double grad_amplification;
  double tbr_amplification;
  double total_tbr_multiplier;
  double coll_tbr_rate;
  double v_reduction;
  double grad_sign_thresh;
  double threshold_std;
  double grad_sign_dist;
  double grad_dependence;
  double rho0;
  double Qm_cutoff;
  double larger_cutoff;
  double upper_mem_thresh;
  double lower_mem_thresh;
  double fk;
  double threshold_decay_rate;
  double grad_decay_rate;
  int HOMOGENEOUS;
  int GAUSSIAN;
  int LINEAR;
  int LARGE;
  int SYMMETRIZED;
  int EXPERIMENT;
  int SAME_PEAK;
  int TRUE_HMG;
  int PBC;
  int PUNCTIFORM;
  int WCA;
  int NOCENTER;
  int power_input;
  int cutoff_multiplier;
  int Nblock;
  int Nlevel;
  int NperCore;
  int Nsim;
  int print_skip;
  int TBR_I;
  int TBR_THETA;
  int TBR_DIST;
  int COLLISIONS;
  int INDEPENDENT;
  int DELAY;
  int time_mem;
  int rec_jump;
  int S_model;
  double S_tau;
  double S_gamma;
  double S_st;
  double S_stot;
  double nocenter_rad;
  int ABP_model;
  double ABP_torque_fac;
  

  std::string type_name;

  void print_parameters(std::string filename) //prints parameters
  {
    ofstream file_out(filename);

    file_out << Nblock << " " << Nlevel << " " << dt << " " << D << " " << D_r << " " << rho << " " << part_diam << " " << active_speed << " " << HOMOGENEOUS;
    file_out << " " << PBC << " " << WCA << " " << LJcutoff << " " << LJepsilon << " " << LJoffset << " " << print_skip << " " << PUNCTIFORM;
    file_out << " " << GAUSSIAN << " " << EXPERIMENT << " " << power_input << " " << avg_tbr << " " << sub_dist << " " << TBR_I << " " << TBR_THETA << " " << COLLISIONS;
    file_out << " " << grad_amplification << " " << tbr_amplification << " " << coll_tbr_rate << " " << TBR_DIST << " " << v_reduction << " " << grad_sign_thresh; 
    file_out << " " << grad_dependence << " " << Qm_cutoff << " " << grad_sign_dist << " " << LINEAR << " " << upper_mem_thresh << " " << lower_mem_thresh << " " << SYMMETRIZED << " " << time_mem;
    file_out << " " << S_model << " " << S_tau << " " << S_gamma << " " << S_st << " " << S_stot << " " << INDEPENDENT << " " << LARGE << " " << NOCENTER << " " << total_tbr_multiplier;
    file_out << " " << rec_jump << " " << threshold_std << " " << fk << " " << DELAY << " " << threshold_decay_rate << " " << ABP_model << " " << ABP_torque_fac << " " << grad_decay_rate << " " << SAME_PEAK << endl;
    file_out.close();
  }

  void update_parameters() //defines new parameters based on the input ones
  {
    
    active_speed/=part_diam;
    grad_sign_dist/=part_diam;
    kBT/=part_diam*part_diam;   //we rescale the parameters to sigma
    eta*=part_diam;
    sub_ratio=1.0/(2.0*sub_dist);
    gamma=3*eta*PI;
    gamma1=gamma/(1.0-(9.0/16.0)*sub_ratio+(1.0/8.0)*pow(sub_ratio,1.0/3.0));//gamma/(9.0/16.0);
    beta=PI*eta;
    beta1=beta/(1.0-(1.0/8.0)*pow(sub_ratio,1.0/3.0));//beta/(7.0/8.0);
    LJepsilon*=kBT;

    nocenter_rad=NOCENTER*1.0/part_diam;

    //eps_norm=LJepsilon/gamma1;  //we avoid dividing everytime by gamma1
    max_force=144.0/13.0*pow(7.0/26.0,7.0/6.0)*LJepsilon;
    D=kBT/gamma1;
    D_r=kBT/beta1;

    Bt=sqrt(2*D*dt);
    Br=sqrt(2*D_r*dt);

    Nsim=pow(Nblock,Nlevel);

    Qm_cutoff=0;
    rho0=0;
    if (v_reduction==1)
    {
      Qm_cutoff=5;
      rho0=floor((Qm_cutoff+.5)*(Qm_cutoff+.5)*PI*0.9);//close packing density divided by 2
    }

    TRUE_HMG=0;
    if (HOMOGENEOUS>0 && GAUSSIAN==0 && EXPERIMENT==0 && LINEAR==0 && SYMMETRIZED==0 && LARGE==0) //if TRUE_HMG is 1, we skip the calculation of velocities and gradients
      TRUE_HMG=1;
       
    print_skip=pow(Nblock,7+Nlevel-18);//pow(Nblock,7); 7 for a bigdt of 0.1

    if(WCA==0)
      cutoff_multiplier=5;
    else
      cutoff_multiplier=1;

    LJcutoff=pow(2,1.0/6.0)*cutoff_multiplier;
    LJoffset=-LJepsilon*4*pow(LJcutoff,-6)*(pow(LJcutoff,-6)-1);

    larger_cutoff=max(LJcutoff,Qm_cutoff);

    //cout << v_reduction << " " << Qm_cutoff << " " << larger_cutoff << " " << endl;

    type_name="";


    if (HOMOGENEOUS>0)
    {
      type_name+="_hmg";
      if (TRUE_HMG>0)
        type_name+="_avg_tbr"+to_string(avg_tbr);
    }
    else if (GAUSSIAN>0)
      type_name+="_gauss";
    else if (LARGE>0)
      type_name+="_large_ring";
    else if (LINEAR>0)
      type_name+="_lin";
    else if (SYMMETRIZED>0)
      type_name+="_symm";
    else
      type_name+="_ring";

    if (LARGE>0 and SAME_PEAK>0)
      type_name+="_samepeak";

    if (S_model>0)
    {
      type_name+="_Smodel";
      time_mem=0;
      rec_jump=0;
      upper_mem_thresh=0;
      lower_mem_thresh=0;
      grad_sign_thresh=0;
      threshold_std=0;
      grad_sign_dist=0;
      fk=0;
    }

    if (INDEPENDENT>0)
    {
      COLLISIONS=0;
      v_reduction=0;
    }
    
    if (EXPERIMENT>0)
      type_name+="_exp";
    
    if (rho>1.0) //if we ever care about different densities
      type_name+="_multi";

    if (rho>=1.0)
      type_name+="_hd";

    type_name+="_"+to_string(power_input)+"muW";
  }


};

std::vector<std::vector<double>> read_matrix(std::fstream &f, int &R, int &C, double &param_1, double &param_2, double &param_3) {
	f >> R >> C >> param_1 >> param_2 >> param_3;
	std::vector<std::vector<double>> matrix(R);
	for (auto& row: matrix) {
		row.resize(C);
		for (auto& val: row) {
			f >> val;
		}
	}
	return matrix;
}

std::vector<double> read_matrix_tovector(std::fstream &f, int &R, int &C, double &param_1, double &param_2, double &param_3) {
	f >> R >> C >> param_1 >> param_2 >> param_3;
	std::vector<double> matrix_tovector(R*C);
  for (auto& val: matrix_tovector) {
		f >> val;
	}
	return matrix_tovector;
}

struct moment{
  valarray<double> D;
  valarray<double> Dvar;
  valarray<double> Dcount; 
};

struct tumbles {
  int sim_number, Nx, Ny, Npart, Ncellx, Ncelly, Nmaxpercell;
  double avg_intensity, avg_tbr, Lx, Ly, pixel_size, rcellx, rcelly;
  parameters par;
  vector<double> tbr_field_vector;
  vector<double> tbr_dist_field_vector;
  //vector<vector<double>> mot_field;

  void read_field(int illumination) //reads the motility field
  {
    int i;
    std::string filename="tbr";
    if (illumination>0)
    {
      filename="ill";
    }
    //cout << "lama" << endl;
    //fstream motfile("mot_fields//mot_field_"+to_string(sim_number)+".dat");
    //mot_field=read_matrix(motfile, Nx, Ny, grain_size, avg_intensity, avg_speed, pixel_size);
    if (par.TRUE_HMG>0)
    {
      avg_tbr=par.avg_tbr;
      pixel_size=1.041667/par.part_diam;
      Nx=1159; //965
      Ny=1159;
      
    }
    else
    {
      std::string tbrfile1_name;
      tbrfile1_name="tbr_fields//"+filename+"_map"+par.type_name+".dat";
      std::cout << tbrfile1_name << endl;

      fstream tbrfile1(tbrfile1_name);
      tbr_field_vector=read_matrix_tovector(tbrfile1, Nx, Ny, avg_intensity, avg_tbr, pixel_size);

      for (i=0;i<Nx*Ny;i++)
      {
        tbr_field_vector[i]=avg_tbr+par.tbr_amplification*(tbr_field_vector[i]-avg_tbr); 
        if (tbr_field_vector[i]<0)
        {
          cout << "Too large tumbling amplification!" << endl;
          tbr_field_vector[i]=0;
        }
      }


      if (par.TBR_DIST>0)
      {
        std::string tbrdistfile_name;
        tbrdistfile_name="tbr_fields//"+filename+"_map_dist"+par.type_name+".dat";
        fstream tbrdistfile(tbrdistfile_name);
        tbr_dist_field_vector=read_matrix_tovector(tbrdistfile, Nx, Ny, avg_intensity, avg_tbr, pixel_size); //careful, we assume that the parameters of both maps (dist and not) are the same

        for (i=0;i<Nx*Ny;i++)
        {
          tbr_dist_field_vector[i]=avg_tbr+par.tbr_amplification*(tbr_dist_field_vector[i]-avg_tbr); 
          if (tbr_dist_field_vector[i]<0)
          {
            cout << "Too large tumbling amplification!" << endl;
            tbr_dist_field_vector[i]=0;
          }
        } 
      }

      pixel_size/=par.part_diam;

    }

    Lx=pixel_size*Nx;
    Ly=pixel_size*Ny;
    Npart=Lx*Ly/(PI*.25)*par.rho;
    cout << "Npart = " << Npart << endl;
    cout << "Nx = " << Nx << endl;
    cout << "Ny = " << Ny << endl;
    Ncellx=floor(Lx/par.larger_cutoff);
    Ncelly=floor(Ly/par.larger_cutoff);
    rcellx=Lx/Ncellx;
    rcelly=Ly/Ncelly;
    Nmaxpercell=(par.larger_cutoff+1)*(par.larger_cutoff+1);
    std::cout << Nx << " " << Ny << " " << avg_intensity << " " << avg_tbr << " " << pixel_size << " " << Npart << " " << rcellx << " " << rcelly << endl;// " " << mot_field[7][5] << 
  }

  void print_tbr_parameters(std::string filename) //prints parameters of the motility field
  {
    ofstream file_out(filename, std::ofstream::out | std::ofstream::app);

    file_out << Nx << " " << Ny;
    file_out << " " << avg_intensity << " " << avg_tbr << " " << pixel_size << " " << Npart << " " << Ncellx << " " << Ncelly << " " << Nmaxpercell <<  endl;
    file_out.close();
  }

};

struct particle {
  mt19937_64 mt;
  normal_distribution<double> norm_distribution;
  uniform_real_distribution<double> uni_distribution;
  valarray<double> x_old;
  valarray<double> y_old;
  valarray<double> trate_old;
  valarray<double> phi_old;
  valarray<double> x;
  valarray<double> y;
  valarray<double> trate;
  valarray<double> torque;
  valarray<double> illval;
  valarray<double> trate_mem;
  valarray<double> v_red;
  valarray<double> phi;
  valarray<int> collision;
  valarray<double> density;
  valarray<double> mem_state;
  valarray<double> distributed_threshold;
  valarray<int> smem_state;
  valarray<double> svar;
  valarray<double> tratex;
  valarray<double> tratey;
  valarray<double> fx;
  valarray<double> fy;
  valarray<double> x_log;
  valarray<double> y_log;
  valarray<double> trate_log;
  valarray<double> phi_log;
  valarray<double> state_log;
  valarray<double> pot;
  valarray<double> vir;

  valarray<double> t_memory;
  
  valarray<double> particle_cell_x;
  valarray<double> particle_cell_y;
  valarray<int> cell_list;
  valarray<int> stuck;
  valarray<int> unstuck;
  parameters par;
  tumbles tbr;
  tumbles ill;

  void initialize_thresholds()
  {
    int i;

    if (par.threshold_std>0)
    {
      for(i=0;i<tbr.Npart; i++)
      {
        distributed_threshold[i]=par.grad_sign_thresh*(1+norm_distribution(mt)*par.threshold_std);
        if  (distributed_threshold[i]<0)
        {
          cout << "Error: gradient threshold too low" << endl;
          distributed_threshold[i]=0;
        }
      }
    }
  }

  void initialize_positions() //initialization of positions on a grid
  {
    int count, i, j, Npartx, Nparty;
    double rate=tbr.Lx/tbr.Ly, dx, dy;    
    
    if (tbr.Npart==1)
    {
      x[0]=(uni_distribution(mt)-.5)*tbr.Lx;
      y[0]=(uni_distribution(mt)-.5)*tbr.Ly;
      if (par.NOCENTER>0)
      {
        while (sqrt(x[0]*x[0]+y[0]*y[0])<par.nocenter_rad)
        {
          x[0]=(uni_distribution(mt)-.5)*tbr.Lx;
          y[0]=(uni_distribution(mt)-.5)*tbr.Ly;
        }
      }
    }
    else
    {
      Npartx=ceil(sqrt(tbr.Npart*rate));
      Nparty=ceil(sqrt(tbr.Npart/rate));

      if (par.NOCENTER>0)
      {
        Npartx=ceil(Npartx*tbr.Lx/sqrt(tbr.Lx*tbr.Lx-par.nocenter_rad*par.nocenter_rad*PI));
        Nparty=ceil(Nparty*tbr.Ly/sqrt(tbr.Ly*tbr.Ly-par.nocenter_rad*par.nocenter_rad*PI));
      }


      dx=tbr.Lx/Npartx;
      dy=tbr.Ly/Nparty;


      count=-1;
      while (count<tbr.Npart)
      {
        for(i=0;i<Npartx; i++)
        {
          for(j=0;j<Nparty;j++)
          {
            count++;
            if(count>=tbr.Npart)
              break; 
            x[count]=(dx-tbr.Lx)*.5+dx*i;
            y[count]=(dy-tbr.Ly)*.5+dy*j;

            if (par.NOCENTER>0 && sqrt(x[count]*x[count]+y[count]*y[count])<par.nocenter_rad)
            {
              count--;
              continue;
            }   
          }
        }
      }
    }
  }

  void create_cell_list() //creation of neighbor cell lists
  {
    int k, l, nx, ny, list_element, pcx, pcy;
    for(list_element=0; list_element<tbr.Nmaxpercell*tbr.Ncellx*tbr.Ncelly; list_element++)
    {
      cell_list[list_element]=-1;
    }
    for(k=0;k<tbr.Npart;k++)
    {
      pcx=floor((x[k]+tbr.Lx*.5)/tbr.rcellx);
      pcy=floor((y[k]+tbr.Ly*.5)/tbr.rcelly);
      if(pcx<0 || pcx>=tbr.Ncellx)
      {
        std::cout << "Particle " << k << " escaped along x!" << endl; 
        abort();
      }
      if(pcy<0 || pcy>=tbr.Ncelly)
      {
        std::cout << "Particle " << k << " escaped along y!"<< endl;
        abort();
      }
      particle_cell_x[k]=pcx;
      particle_cell_y[k]=pcy;

      for(l=0;l<tbr.Nmaxpercell;l++)
      {
        list_element=l+tbr.Nmaxpercell*(pcy+tbr.Ncelly*pcx);
        if(cell_list[list_element]==-1)
        {
          cell_list[list_element]=k;
          break;
        }
        if(l==tbr.Nmaxpercell-1 && cell_list[list_element]!=-1)
        {
          std::cout << "Too many particle per cell!" << endl;
          abort();
        }
      }

      //cout << particle_cell_x[k] << " " << particle_cell_y[k] << endl;
    }
    //for(int c=0;c<mot.Ncellx;c++)
    //  {
    //  for(int b=0;b<mot.Ncellx;b++)
    //  {
    //    cout << c << " "  << b << " " << cell_list[0+mot.Nmaxpercell*(b+mot.Ncelly*c)] << endl;
    //  }
    //}
  }

  void slip_boundary_condition(int time_step) //function that creates a wall on the border
  {
    int k;
    double dxwall, dywall, sigmadr6, potx, poty;
    for(k=0;k<tbr.Npart;k++)
    {
      if(particle_cell_x[k]!=0 && particle_cell_y[k]!=0 && particle_cell_x[k]!=tbr.Ncellx-1 && particle_cell_y[k]!=tbr.Ncelly-1)
        continue;

      dxwall=par.LJcutoff*2;
      dywall=par.LJcutoff*2;

      if(particle_cell_x[k]==0)
        dxwall=(x[k]+tbr.Lx*.5)*2;
      else if(particle_cell_x[k]==tbr.Ncellx-1)
        dxwall=(x[k]-tbr.Lx*.5)*2;

      if(particle_cell_y[k]==0)
        dywall=(y[k]+tbr.Ly*.5)*2;
      else if(particle_cell_y[k]==tbr.Ncelly-1)
        dywall=(y[k]-tbr.Ly*.5)*2;

      if(fabs(dxwall)<par.LJcutoff)
      {
        sigmadr6=pow(dxwall,-6);
        fx[k]+=24*par.LJepsilon/dxwall*sigmadr6*(2*sigmadr6-1);
        pot[time_step]+=par.LJepsilon*4*sigmadr6*(sigmadr6-1)+par.LJoffset;
        vir[time_step]+=dxwall*fx[k];
      }
      if(fabs(dywall)<par.LJcutoff)
      {
        sigmadr6=pow(dywall,-6);
        fy[k]+=24*par.LJepsilon/dywall*sigmadr6*(2*sigmadr6-1);
        pot[time_step]+=par.LJepsilon*4*sigmadr6*(sigmadr6-1)+par.LJoffset;
        vir[time_step]+=dywall*fy[k];
      }
   
    }

    vir[time_step]/=(2*tbr.Lx*tbr.Ly);
  }

  void periodic_boundary_conditions() //simple periodic boundary conditions
  {
    int k;
    for(k=0;k<tbr.Npart;k++)
    {
      x[k]-=rint(x[k]/tbr.Lx)*tbr.Lx;
      y[k]-=rint(y[k]/tbr.Ly)*tbr.Ly;
    }
  }

  double tbr_orientation_cube(double theta, double dist2, int mem, int smem, double grad_decay)
  {
    double diff, sign_change, sign_change_dist;
    diff=theta-1.48989774;
    sign_change_dist=1.0;
    //cout << tbrate << endl;
    //cout << tbr.avg_tbr << endl;
    if (par.grad_sign_dist>0 and dist2<par.grad_sign_dist*par.grad_sign_dist)
      sign_change_dist=-1.0;
    return (1.+smem*mem*sign_change_dist*par.grad_amplification*0.01413336*diff*diff*diff*grad_decay);//the one with 0.014 is the experimental one
  }

  double tbr_orientation(double theta, double dist2, double mem, int smem, double grad_decay)
  {
    double diff, sign_change_dist;
    diff=theta-1.44109913;
    sign_change_dist=1.0;
    //cout << tbrate << endl;
    //cout << tbr.avg_tbr << endl;
    if (par.grad_sign_dist>0 and dist2<par.grad_sign_dist*par.grad_sign_dist)
      sign_change_dist=-1.0;
    return (1.+smem*mem*sign_change_dist*par.grad_amplification*0.01887769*diff*grad_decay);//the one with 0.014 is the experimental one
  }

  void compute_trates() //computation of tumbling rates
  {
    int nx, ny, nxright, nxleft, nyup, nydown, xcells, ycells, i, j, k, counter, nxi, nyj, nxileft, nxiright, nyjdown, nyjup;
    double cell_dx, cell_dy, radius, trate_pot, tratex_fx, tratey_fy;
    for(k=0;k<tbr.Npart;k++)
    {
      trate_pot=0;
      tratex_fx=0;
      tratey_fy=0;
      if (par.TRUE_HMG==1)
      {
        trate_pot=tbr.avg_tbr;
        tratex_fx=0;
        tratey_fy=0;
      }
      else
      {
        if (par.PUNCTIFORM==1)
        {
          nx=floor((x[k]+tbr.Lx*.5)/tbr.pixel_size);
          ny=floor((y[k]+tbr.Ly*.5)/tbr.pixel_size);
          nxright=nx+1;
          nxleft=nx-1;
          nyup=ny+1;
          nydown=ny-1;

          if (nxleft<0)
            nxleft+=tbr.Nx;
          if (nxright>=tbr.Nx)
            nxright-=tbr.Nx;
          if (nydown<0)
            nydown+=tbr.Ny;
          if (nyup>=tbr.Ny)
            nyup-=tbr.Ny;

          trate_pot=tbr.tbr_field_vector[nx*tbr.Ny+ny];
          tratey_fy=(tbr.tbr_field_vector[tbr.Ny*nx+nyup]-tbr.tbr_field_vector[tbr.Ny*nx+nydown])/(2*tbr.pixel_size);
          tratex_fx=(tbr.tbr_field_vector[tbr.Ny*nxright+ny]-tbr.tbr_field_vector[tbr.Ny*nxleft+ny])/(2*tbr.pixel_size);

        }
        else
        {
          radius=.5;
          nx=floor((x[k]+tbr.Lx*.5)/tbr.pixel_size);
          ny=floor((y[k]+tbr.Ly*.5)/tbr.pixel_size);
          nxright=floor((x[k]+tbr.Lx*.5+radius)/tbr.pixel_size);
          nxleft=floor((x[k]+tbr.Lx*.5-radius)/tbr.pixel_size);
          nyup=floor((y[k]+tbr.Ly*.5+radius)/tbr.pixel_size);
          nydown=floor((y[k]+tbr.Ly*.5-radius)/tbr.pixel_size);
          xcells=nxright-nxleft;
          ycells=nyup-nydown;
          //cout  << nx << " " << ny << " " << nxright << " " << nxleft << " " << nyup << " " << nydown << endl;
          counter=0;
          for (i=0; i<xcells; i++)
          {
            nxi=i+nxleft;
            if(nxi<0)
              nxi+=tbr.Nx;
            if(nxi>=tbr.Nx)
              nxi-=tbr.Nx;

            for (j=0; j<ycells; j++)
            {
              nyj=j+nydown;
              if(nyj<0)
                nyj+=tbr.Ny;
              if(nyj>=tbr.Ny)
                nyj-=tbr.Ny;
              cell_dx=tbr.pixel_size*(nxleft+i)-tbr.Lx*.5-x[k];           
              cell_dy=tbr.pixel_size*(nydown+j)-tbr.Ly*.5-y[k];
              //cout << "step=" << time_step << " " << cell_dx << " " << cell_dy << " " << cell_dx*cell_dx+cell_dy*cell_dy-radius*radius << " " << cell_dx*cos(phi[time_step])+cell_dy*sin(phi[time_step]) << endl;
              if (cell_dx*cell_dx+cell_dy*cell_dy<radius*radius)
              {
                counter++;
                trate_pot+=(tbr.tbr_field_vector[nxi*tbr.Ny+nyj]-trate_pot)/counter;//needs to be checked
              }
              
              //else
              //{
              //  if (cell_dx*cell_dx+cell_dy*cell_dy<radius*radius && (cell_dx*cos(phi[k])+cell_dy*sin(phi[k]))*par.invert_factor<0)
              //  {
              //    counter++;
              //    v_pot+=(mot.mot_field_vector[nxi*mot.Ny+nyj]-v_pot)/counter;//needs to be checked
              //  }
              //}
            }     
          }
          //cout << counter << endl;
          //cout << cell_dx << " " << counter << " " << radius << " " << v[time_step] << endl;
          for (i=0; i<xcells; i++)
          {
            nyjdown=nydown;
            nyjup=nyup;
            nxi=i+nxleft;
            if(nxi<0)
              nxi+=tbr.Nx;
            if(nxi>=tbr.Nx)
              nxi-=tbr.Nx;
            if(nyjdown<0)
              nyjdown+=tbr.Ny;
            if(nyjup>=tbr.Ny)
              nyjup-=tbr.Ny;
            tratey_fy+=tbr.tbr_field_vector[tbr.Ny*nxi+nyjup]-tbr.tbr_field_vector[tbr.Ny*nxi+nyjdown];
          }
          for (j=0; j<ycells; j++)
          {
            nxileft=nxleft;
            nxiright=nxright;
            nyj=j+nydown;
            if(nyj<0)
              nyj+=tbr.Ny;
            if(nyj>=tbr.Ny)
              nyj-=tbr.Ny;
            if(nxileft<0)
              nxileft+=tbr.Nx;
            if(nxiright>=tbr.Nx)
              nxiright-=tbr.Nx;
            tratex_fx+=tbr.tbr_field_vector[tbr.Ny*nxiright+nyj]-tbr.tbr_field_vector[tbr.Ny*nxileft+nyj];
          }
          tratey_fy/=xcells;
          tratex_fx/=ycells;
          //vx[time_step]=(mot.mot_field[nxright][ny]-mot.mot_field[nxleft][ny])/(part_diam); //where we don't average on the torque
          //vy[time_step]=(mot.mot_field[nx][nyup]-mot.mot_field[nx][nydown])/(part_diam);
        }
      }
      //cout << k << " " << x[k+time_step*mot.Npart] << " " << y[k+time_step*mot.Npart] << " " << phi[k+time_step*mot.Npart] << " " << v[k+time_step*mot.Npart] << " " << vx[k+time_step*mot.Npart] << " " << vy[k+time_step*mot.Npart] << endl;

      trate[k]=trate_pot;
      tratey[k]=tratey_fy;
      tratex[k]=tratex_fx;
    } 
  }

  void compute_trates_dist() //computation of tumbling rates
  {
    int nx, ny, nxright, nxleft, nyup, nydown, xcells, ycells, i, j, k, counter, nxi, nyj, nxileft, nxiright, nyjdown, nyjup;
    double cell_dx, cell_dy, radius, trate_pot, tratex_fx, tratey_fy;
    for(k=0;k<tbr.Npart;k++)
    {
      trate_pot=0;
      tratex_fx=0;
      tratey_fy=0;
      if (par.TRUE_HMG==1)
      {
        trate_pot=tbr.avg_tbr;
        tratex_fx=0;
        tratey_fy=0;
      }
      else
      {
        if (par.PUNCTIFORM==1)
        {
          nx=floor((x[k]+tbr.Lx*.5)/tbr.pixel_size);
          ny=floor((y[k]+tbr.Ly*.5)/tbr.pixel_size);
          nxright=nx+1;
          nxleft=nx-1;
          nyup=ny+1;
          nydown=ny-1;

          if (nxleft<0)
            nxleft+=tbr.Nx;
          if (nxright>=tbr.Nx)
            nxright-=tbr.Nx;
          if (nydown<0)
            nydown+=tbr.Ny;
          if (nyup>=tbr.Ny)
            nyup-=tbr.Ny;

          trate_pot=tbr.tbr_dist_field_vector[nx*tbr.Ny+ny];
          tratey_fy=(tbr.tbr_field_vector[tbr.Ny*nx+nyup]-tbr.tbr_field_vector[tbr.Ny*nx+nydown])/(2*tbr.pixel_size);
          tratex_fx=(tbr.tbr_field_vector[tbr.Ny*nxright+ny]-tbr.tbr_field_vector[tbr.Ny*nxleft+ny])/(2*tbr.pixel_size);

        }
        else
        {
          radius=.5;
          nx=floor((x[k]+tbr.Lx*.5)/tbr.pixel_size);
          ny=floor((y[k]+tbr.Ly*.5)/tbr.pixel_size);
          nxright=floor((x[k]+tbr.Lx*.5+radius)/tbr.pixel_size);
          nxleft=floor((x[k]+tbr.Lx*.5-radius)/tbr.pixel_size);
          nyup=floor((y[k]+tbr.Ly*.5+radius)/tbr.pixel_size);
          nydown=floor((y[k]+tbr.Ly*.5-radius)/tbr.pixel_size);
          xcells=nxright-nxleft;
          ycells=nyup-nydown;
          //cout  << nx << " " << ny << " " << nxright << " " << nxleft << " " << nyup << " " << nydown << endl;
          counter=0;
          for (i=0; i<xcells; i++)
          {
            nxi=i+nxleft;
            if(nxi<0)
              nxi+=tbr.Nx;
            if(nxi>=tbr.Nx)
              nxi-=tbr.Nx;

            for (j=0; j<ycells; j++)
            {
              nyj=j+nydown;
              if(nyj<0)
                nyj+=tbr.Ny;
              if(nyj>=tbr.Ny)
                nyj-=tbr.Ny;
              cell_dx=tbr.pixel_size*(nxleft+i)-tbr.Lx*.5-x[k];           
              cell_dy=tbr.pixel_size*(nydown+j)-tbr.Ly*.5-y[k];
              //cout << "step=" << time_step << " " << cell_dx << " " << cell_dy << " " << cell_dx*cell_dx+cell_dy*cell_dy-radius*radius << " " << cell_dx*cos(phi[time_step])+cell_dy*sin(phi[time_step]) << endl;
              if (cell_dx*cell_dx+cell_dy*cell_dy<radius*radius)
              {
                counter++;
                trate_pot+=(tbr.tbr_dist_field_vector[nxi*tbr.Ny+nyj]-trate_pot)/counter;//needs to be checked
              }
              
              //else
              //{
              //  if (cell_dx*cell_dx+cell_dy*cell_dy<radius*radius && (cell_dx*cos(phi[k])+cell_dy*sin(phi[k]))*par.invert_factor<0)
              //  {
              //    counter++;
              //    v_pot+=(mot.mot_field_vector[nxi*mot.Ny+nyj]-v_pot)/counter;//needs to be checked
              //  }
              //}
            }     
          }
          //cout << counter << endl;
          //cout << cell_dx << " " << counter << " " << radius << " " << v[time_step] << endl;
          for (i=0; i<xcells; i++)
          {
            nyjdown=nydown;
            nyjup=nyup;
            nxi=i+nxleft;
            if(nxi<0)
              nxi+=tbr.Nx;
            if(nxi>=tbr.Nx)
              nxi-=tbr.Nx;
            if(nyjdown<0)
              nyjdown+=tbr.Ny;
            if(nyjup>=tbr.Ny)
              nyjup-=tbr.Ny;
            tratey_fy+=tbr.tbr_field_vector[tbr.Ny*nxi+nyjup]-tbr.tbr_field_vector[tbr.Ny*nxi+nyjdown];
          }
          for (j=0; j<ycells; j++)
          {
            nxileft=nxleft;
            nxiright=nxright;
            nyj=j+nydown;
            if(nyj<0)
              nyj+=tbr.Ny;
            if(nyj>=tbr.Ny)
              nyj-=tbr.Ny;
            if(nxileft<0)
              nxileft+=tbr.Nx;
            if(nxiright>=tbr.Nx)
              nxiright-=tbr.Nx;
            tratex_fx+=tbr.tbr_field_vector[tbr.Ny*nxiright+nyj]-tbr.tbr_field_vector[tbr.Ny*nxileft+nyj];
          }
          tratey_fy/=xcells;
          tratex_fx/=ycells;
          //vx[time_step]=(mot.mot_field[nxright][ny]-mot.mot_field[nxleft][ny])/(part_diam); //where we don't average on the torque
          //vy[time_step]=(mot.mot_field[nx][nyup]-mot.mot_field[nx][nydown])/(part_diam);
        }
      }
      //cout << k << " " << x[k+time_step*mot.Npart] << " " << y[k+time_step*mot.Npart] << " " << phi[k+time_step*mot.Npart] << " " << v[k+time_step*mot.Npart] << " " << vx[k+time_step*mot.Npart] << " " << vy[k+time_step*mot.Npart] << endl;

      trate[k]=trate_pot;
      tratey[k]=tratey_fy;
      tratex[k]=tratex_fx;
    } 
  }

  void compute_illval() //computation of tumbling rates
  {
    int nx, ny, nxright, nxleft, nyup, nydown, xcells, ycells, i, j, k, counter, nxi, nyj, nxileft, nxiright, nyjdown, nyjup;
    double cell_dx, cell_dy, radius, ill_pot;
    for(k=0;k<ill.Npart;k++)
    {
      ill_pot=0;
      if (par.TRUE_HMG==1)
      {
        ill_pot=ill.avg_intensity;
      }
      else
      {
        if (par.PUNCTIFORM==1)
        {
          nx=floor((x[k]+ill.Lx*.5)/ill.pixel_size);
          ny=floor((y[k]+ill.Ly*.5)/ill.pixel_size);

          ill_pot=ill.tbr_field_vector[nx*ill.Ny+ny];

        }
        else
        {
          radius=.5;
          nx=floor((x[k]+ill.Lx*.5)/ill.pixel_size);
          ny=floor((y[k]+ill.Ly*.5)/ill.pixel_size);
          nxright=floor((x[k]+ill.Lx*.5+radius)/ill.pixel_size);
          nxleft=floor((x[k]+ill.Lx*.5-radius)/ill.pixel_size);
          nyup=floor((y[k]+ill.Ly*.5+radius)/ill.pixel_size);
          nydown=floor((y[k]+ill.Ly*.5-radius)/ill.pixel_size);
          xcells=nxright-nxleft;
          ycells=nyup-nydown;
          //cout  << nx << " " << ny << " " << nxright << " " << nxleft << " " << nyup << " " << nydown << endl;
          counter=0;
          for (i=0; i<xcells; i++)
          {
            nxi=i+nxleft;
            if(nxi<0)
              nxi+=ill.Nx;
            if(nxi>=ill.Nx)
              nxi-=ill.Nx;

            for (j=0; j<ycells; j++)
            {
              nyj=j+nydown;
              if(nyj<0)
                nyj+=ill.Ny;
              if(nyj>=ill.Ny)
                nyj-=ill.Ny;
              cell_dx=ill.pixel_size*(nxleft+i)-ill.Lx*.5-x[k];           
              cell_dy=ill.pixel_size*(nydown+j)-ill.Ly*.5-y[k];
              //cout << "step=" << time_step << " " << cell_dx << " " << cell_dy << " " << cell_dx*cell_dx+cell_dy*cell_dy-radius*radius << " " << cell_dx*cos(phi[time_step])+cell_dy*sin(phi[time_step]) << endl;
              if (cell_dx*cell_dx+cell_dy*cell_dy<radius*radius)
              {
                counter++;
                ill_pot+=(ill.tbr_field_vector[nxi*ill.Ny+nyj]-ill_pot)/counter;//needs to be checked
              }

            }     
          }

        }
      }
      //cout << k << " " << x[k+time_step*mot.Npart] << " " << y[k+time_step*mot.Npart] << " " << phi[k+time_step*mot.Npart] << " " << v[k+time_step*mot.Npart] << " " << vx[k+time_step*mot.Npart] << " " << vy[k+time_step*mot.Npart] << endl;

      illval[k]=ill_pot;
    } 
  }

  void reset_trates()
  {
    for(int k=0;k<tbr.Npart;k++)
    {
      trate[k]=par.avg_tbr;
    }
  }


  void set_memory(int timestep)
  {
    double thresh_decay = 1;
    if (par.threshold_decay_rate>0.0)
      thresh_decay=exp(-timestep*par.dt*par.threshold_decay_rate);
    if (par.threshold_decay_rate<0.0)
      thresh_decay=2-exp(timestep*par.dt*par.threshold_decay_rate);
    if (par.fk>0)
    {
      for(int k=0;k<tbr.Npart;k++)
      {
        mem_state[k]=2.0/(1.0+exp((trate[k]-distributed_threshold[k]*thresh_decay)/par.fk))-1.0;
      }
    }
    else
    {
      for(int k=0;k<tbr.Npart;k++)
      {
        //cout << trate[k] << " " << (trate[k]<par.lower_mem_thresh) << " " << (par.lower_mem_thresh) << " " << (par.upper_mem_thresh) << endl;
        if (trate[k]>distributed_threshold[k]*thresh_decay)
        {
          mem_state[k]=-1.0;
        }
        else
        {
          mem_state[k]=1.0;
        }
        if (par.upper_mem_thresh>0)
        {
          if (trate[k]>=par.upper_mem_thresh)
          {
            mem_state[k]=-1.0;
          }
          if (trate[k]<par.lower_mem_thresh)
          {
            mem_state[k]=1.0;
          }
        }
      }
    }
  }

  void collision_trates()
  {
    for(int k=0;k<tbr.Npart;k++)
    {
      if (collision[k]==1)
        trate[k]=par.coll_tbr_rate;
    }
  }
  
  void collision_speeds()
  {
    for(int k=0;k<tbr.Npart;k++)
    {
      v_red[k]=1.0-density[k]/par.rho0;
      //if (v_red[k]<0.05)
      //  cout << v_red[k] << endl;
      if (v_red[k]<0) v_red[k]=0;///to avoid negative speeds
    }
  }

  void compute_trates_theta(double timestep)
  {
    double theta, dist2;
    double grad_decay = 1;
    if (par.grad_decay_rate>0.0)
      grad_decay=exp(-timestep*par.dt*par.grad_decay_rate);

    for(int k=0;k<tbr.Npart;k++)
    {
      if (tratey[k]!=0 or tratex[k]!=0)
      {
        theta=atan2(tratey[k],tratex[k])-phi[k];
        theta-=rint(theta/(2*PI))*2*PI;
        if (par.LINEAR)
        {
          dist2=x[k]*x[k];
        }
        else
        {
          dist2=x[k]*x[k]+y[k]*y[k];
        }
        trate[k]*= tbr_orientation(fabs(theta), dist2, mem_state[k], smem_state[k],grad_decay);
        if (par.grad_dependence>0)
        {
          //if (sqrt(x[k]*x[k]+y[k]*y[k])>25)//remove
          //  trate[k]*= tbr_orientation(fabs(theta), x[k]*x[k]+y[k]*y[k]);//remove
          trate[k] *= par.grad_dependence*sqrt(tratey[k]*tratey[k]+tratex[k]*tratex[k])*1000;
        }
      }
    }

  }

  void tumbling_handler(int timestep)
  {
    if (par.TBR_THETA==1 || par.TBR_I==1)
    {
      if (par.TBR_DIST==0)
        compute_trates();
      else
        compute_trates_dist();
      if (par.time_mem>0)
      {
        if (par.DELAY==1)
          tumbling_delay(timestep);
        else
          tumbling_memory(timestep);
      }
      if (par.grad_sign_thresh>0 or par.upper_mem_thresh>0)
        set_memory(timestep);
      if (par.TBR_I==0)
        reset_trates();
      if (par.TBR_THETA==1)
        compute_trates_theta(timestep); 
    }
    if (par.COLLISIONS==1)
      collision_trates();
    
    if (par.S_model==1)
      compute_illval();
  }

  void torque_handler(int timestep)
  {
    double tnorm;
    compute_trates();

    if (par.time_mem>0)
    {
      if (par.DELAY==1)
        tumbling_delay(timestep);
      else
        tumbling_memory(timestep);
    }
    if (par.grad_sign_thresh>0 or par.upper_mem_thresh>0)
      set_memory(timestep);

    for(int k=0;k<tbr.Npart;k++)
    {
      //tnorm=sqrt(tratey[k]*tratey[k]+tratex[k]*tratex[k]);
      torque[k]=smem_state[k]*mem_state[k]*par.ABP_torque_fac*(cos(phi[k])*tratey[k]-sin(phi[k])*tratex[k]);// /tnorm
      //cout << torque[k] << endl;
    }
    
    if (par.S_model==1)
      compute_illval();
  }

  void tumbling_memory(int time_step)
  {
    int k, time_idx, first_passage;
    time_idx=time_step%par.time_mem;
    first_passage=(time_step/par.time_mem==0);
    for(k=0;k<tbr.Npart;k++)
    {
      t_memory[k+time_idx*tbr.Npart]=trate[k];
      if (first_passage)
      {
        trate_mem[k]+=(trate[k]-trate_mem[k])/(time_idx+1);
      }
      else
      {
        trate_mem[k]+=(trate[k]-t_memory[k+((time_idx+1)%par.time_mem)*tbr.Npart])/par.time_mem;
      }
      trate[k]=trate_mem[k];        
    }
  }

  void tumbling_delay(int time_step)
  {
    int k, time_idx, first_passage;
    time_idx=time_step%par.time_mem;
    first_passage=(time_step/par.time_mem==0);
    for(k=0;k<tbr.Npart;k++)
    {
      t_memory[k+time_idx*tbr.Npart]=trate[k];
      if (first_passage)
      {
        trate[k]=t_memory[k];
      }
      else
      {
        trate[k]=t_memory[k+(time_idx+1)*tbr.Npart];
      }      
    }
  }

  void compute_forces(int time_step) //computation of inter-particle forces
  {
    int i, j, k, l, nx, ny, part2;
    double dx12, dy12, dr2, sigmadr6, fbase;
    for(k=0;k<tbr.Npart;k++)
    {
      nx=particle_cell_x[k];
      ny=particle_cell_y[k];
      for(i=nx-1;i<=nx+1;i++)
      {
        if(i<0 || i>=tbr.Ncellx)
          continue;
        for(j=ny-1;j<=ny+1;j++)
        {
          if(j<0 || j>=tbr.Ncelly)
            continue;
          
          for(l=0;l<tbr.Nmaxpercell;l++)
          {
            part2=cell_list[l+tbr.Nmaxpercell*(j+tbr.Ncelly*i)];
            if(part2==-1)
              break;
            if(part2>=k)
              continue;
            
            dx12=x[k]-x[part2];
            dy12=y[k]-y[part2];
            dr2=pow(dx12,2)+pow(dy12,2);
            if (dr2<par.Qm_cutoff*par.Qm_cutoff)
            {
              density[k] += 1;
              density[part2] += 1;
            }
            if(dr2>par.LJcutoff*par.LJcutoff)
              continue;
            if (dr2<1.)
            {
              collision[k] = 1;
              collision[part2] = 1;
            }
            sigmadr6=pow(dr2,-3);
            fbase=24*par.LJepsilon/dr2*sigmadr6*(2*sigmadr6-1);
            fx[k]+=fbase*dx12;
            fy[k]+=fbase*dy12;
            fx[part2]-=fbase*dx12;
            fy[part2]-=fbase*dy12;
            pot[time_step]+=par.LJepsilon*4*sigmadr6*(sigmadr6-1)+par.LJoffset;
            vir[time_step]+=fx[k]*dx12+fy[k]*dy12;
            
          }
          
        }
      }
    }
    vir[time_step]/=2*tbr.Lx*tbr.Ly;
    
  }

  void compute_forces_PBC(int time_step) //computation of inter-particle forces with PBC 
  {
    int i, j, k, l, nx, ny, part2, part2nx, part2ny;
    double dx12, dy12, dr2, dr2i, sigmadr6, fbase, dfx, dfy;


    for(k=1;k<tbr.Npart;k++)
    {
      nx=particle_cell_x[k];
      ny=particle_cell_y[k];
      for(i=nx-1;i<=nx+1;i++)
      {
        part2nx=i;
        if(i<0)
        {
          part2nx=tbr.Ncellx-1;
          if (part2nx==nx)
            continue;
        }

        else if(i>=tbr.Ncellx)
        {
          part2nx=0;
          if (part2nx==nx)
            continue;
        }

        
        for(j=ny-1;j<=ny+1;j++)
        {
          part2ny=j;
          if(j<0)
          {
            part2ny=tbr.Ncelly-1;
            if (part2ny==ny)
              continue;
          }

          else if(j>=tbr.Ncelly)
          {
            part2ny=0;
            if (part2ny==ny)
              continue;
          }

          for(l=0;l<tbr.Nmaxpercell;l++)
          {
            part2=cell_list[l+tbr.Nmaxpercell*(part2ny+tbr.Ncelly*part2nx)];
            if(part2==-1)
              break;
            if(part2>=k)
              continue;
            dx12=x[k]-x[part2];
            dy12=y[k]-y[part2];
            dx12-=rint(dx12/tbr.Lx)*tbr.Lx;
            dy12-=rint(dy12/tbr.Ly)*tbr.Ly;
            dr2=dx12*dx12+dy12*dy12;
            //if(dr2>pow(tbr.Ly/tbr.Ncelly*2,2.0)*2)
            //  cout << dr2 << endl;
            dr2i=1.0/dr2;
            if (dr2<par.Qm_cutoff*par.Qm_cutoff)
            {
              //cout << sqrt(dr2) << endl;
              density[k] += 1;
              density[part2] += 1;
            }
            if(dr2>par.LJcutoff*par.LJcutoff)
              continue;
            if (dr2<1.)
            {
              collision[k] = 1;
              collision[part2] = 1;
            }
            sigmadr6=dr2i*dr2i*dr2i;
            fbase=24*par.LJepsilon*dr2i*sigmadr6*(2*sigmadr6-1);
            dfx=fbase*dx12;
            dfy=fbase*dy12;
            fx[k]+=dfx;
            fy[k]+=dfy;
            fx[part2]-=dfx;
            fy[part2]-=dfy;
            pot[time_step]+=par.LJepsilon*4*sigmadr6*(sigmadr6-1)+par.LJoffset;
            vir[time_step]+=fx[k]*dx12+fy[k]*dy12;

            //if (time_step>1030 && time_step<1050 && (k==5||k==2))
            //  {
            //    cout << time_step << endl;
            //    cout << k << " " << particle_cell_x[k] << " " << particle_cell_y[k] << " " << cell_list[0+tbr.Nmaxpercell*(particle_cell_y[k]+tbr.Ncelly*particle_cell_x[k])] << " " << cell_list[1+tbr.Nmaxpercell*(particle_cell_y[k]+tbr.Ncelly*particle_cell_x[k])] << endl;
            //    cout << part2 << " " << particle_cell_x[part2] << " " << particle_cell_y[part2] << " " << cell_list[0+tbr.Nmaxpercell*(particle_cell_y[part2]+tbr.Ncelly*particle_cell_x[part2])] << " " << cell_list[1+tbr.Nmaxpercell*(particle_cell_y[part2]+tbr.Ncelly*particle_cell_x[part2])] << endl;
            //    cout << sqrt(dr2) << " " << fbase << " " << fx[k] << " " << fy[k] << endl;
            //  }
            
            //if(fbase<=0.0)
            //  cout << dr2 << endl;
            //if(dr2<1.0)
            //  cout << x[k] << " " << x[part2] << " " << dx12 << " " << y[k] << " " << y[part2] << " " << dy12 << endl;
            // cout << par.eps_norm << " " << sqrt(dr2) << " " << 24*dr2i*sigmadr6*(2*sigmadr6-1) << " " << fbase << " " << dr2i << endl;
          }
          
        }
      }
    }
    vir[time_step]/=2*tbr.Lx*tbr.Ly;
    //if (time_step%1000==0)
    //{
    //  cout << time_step << " " << pot[time_step] << " " << vir[time_step] << endl; 
    //}
  }

  void clean_forces_trategradients() //cleans gradients and forces vector
  {
    int k;
    for (k=0;k<tbr.Npart;k++)
    {
      fx[k]=0;
      fy[k]=0;
      tratex[k]=0;
      tratey[k]=0;
      collision[k]=0;
      density[k]=0;
      torque[k]=0;
    }
  }

  void save_logs(int time_step) //copies coordinates velocity and forces on a log file only at certain timesteps
  {
    int k, time_el;
    if (time_step%par.print_skip==0)
    {
      time_el=time_step/par.print_skip;
      for(k=0;k<tbr.Npart;k++)
      {
        x_log[k+time_el*tbr.Npart]=x[k];
        y_log[k+time_el*tbr.Npart]=y[k];
        phi_log[k+time_el*tbr.Npart]=phi[k];
        trate_log[k+time_el*tbr.Npart]=trate[k];
        state_log[k+time_el*tbr.Npart]=mem_state[k]*smem_state[k];
      }
      //cout << time_step/par.print_skip << endl;
    } 
  }
};

struct simulation {
  mt19937_64 mt;
  normal_distribution<double> norm_distribution;
  uniform_real_distribution<double> uni_distribution;
  parameters par;
  tumbles tbr_all;
  tumbles ill_all;
  int sim_number;
  particle part;
  particle newpart;
  tumbles tbr;
  tumbles ill;
  moment M1;
  moment M2;
  tumbles make_tumbles()
  {
    tumbles ret;
    ret=tbr_all;
    return ret;
  }
  tumbles make_ill()
  {
    tumbles ret;
    ret=ill_all;
    return ret;
  }

  moment make_moment()
  {
    moment ret;
    ret.D=valarray<double> (0.0,par.Nlevel+1);
    ret.Dvar=valarray<double> (0.0,par.Nlevel+1);
    ret.Dcount=valarray<double> (0.0,par.Nlevel+1);
    return ret;
  }

  particle make_particle()
  {
    particle ret;
    ret.mt=mt;
    ret.norm_distribution=norm_distribution;
    ret.uni_distribution=uni_distribution;

    ret.x_old = valarray<double> (0.0,tbr.Npart);
    ret.y_old = valarray<double> (0.0,tbr.Npart);
    ret.trate_old = valarray<double> (0.0,tbr.Npart);
    ret.phi_old = valarray<double> (0.0,tbr.Npart);

    ret.x = valarray<double> (0.0,tbr.Npart);
    ret.y = valarray<double> (0.0,tbr.Npart);
    ret.trate = valarray<double> (0.0,tbr.Npart);
    ret.illval = valarray<double> (0.0,tbr.Npart);
    ret.trate_mem = valarray<double> (0.0,tbr.Npart);
    ret.v_red = valarray<double> (1.0,tbr.Npart);
    ret.collision = valarray<int> (0,tbr.Npart);
    ret.density = valarray<double> (0.0,tbr.Npart);
    ret.mem_state = valarray<double> (1.0,tbr.Npart);
    ret.distributed_threshold = valarray<double> (par.grad_sign_thresh,tbr.Npart);
    ret.smem_state = valarray<int> (1,tbr.Npart);
    ret.svar = valarray<double> (0.0,tbr.Npart);
    ret.phi = valarray<double> (0.0,tbr.Npart);
    ret.torque = valarray<double> (0.0,tbr.Npart);
    ret.tratex = valarray<double> (0.0,tbr.Npart);
    ret.tratey = valarray<double> (0.0,tbr.Npart);
    ret.fx = valarray<double> (0.0,tbr.Npart);
    ret.fy = valarray<double> (0.0,tbr.Npart);
    ret.particle_cell_x = valarray<double> (0.0,tbr.Npart);
    ret.particle_cell_y = valarray<double> (0.0,tbr.Npart);

    ret.t_memory = valarray<double> (0.0,tbr.Npart*par.time_mem);

    ret.x_log = valarray<double> (0.0,tbr.Npart*(par.Nsim/par.print_skip+1));
    ret.y_log = valarray<double> (0.0,tbr.Npart*(par.Nsim/par.print_skip+1));
    ret.trate_log = valarray<double> (0.0,tbr.Npart*(par.Nsim/par.print_skip+1));
    ret.phi_log = valarray<double> (0.0,tbr.Npart*(par.Nsim/par.print_skip+1));
    ret.state_log = valarray<double> (0.0,tbr.Npart*(par.Nsim/par.print_skip+1));
    ret.pot = valarray<double> (0.0,par.Nsim+1);
    ret.vir = valarray<double> (0.0,par.Nsim+1);

    ret.cell_list = valarray<int> (-1,tbr.Ncellx*tbr.Ncelly*tbr.Nmaxpercell);
    ret.par = par;
    ret.tbr = tbr;
    ret.ill = ill;

    ret.initialize_positions();

    ret.initialize_thresholds();

    ret.create_cell_list();

    if (par.PBC==0)
    {
      ret.slip_boundary_condition(0);

      ret.compute_forces(0);
    }
    else
    {
      ret.periodic_boundary_conditions();

      ret.compute_forces_PBC(0);
    }


    for(int k=0;k<tbr.Npart;k++)
    {
      ret.phi[k] = (uni_distribution(mt)-.5)*2*PI;
    }

    ret.tumbling_handler(0);

    ret.torque_handler(0);

    ret.save_logs(0);
   
    return ret;
  }

  void init_sim() //initialize simulation
  {
    tbr = make_tumbles();
    ill = make_ill();
    part = make_particle();
  }

  void md_step_euler(int i) //Euler integration
  {
    int k;
    double dx, dy, ds, Bt, Br, rnum, tprob;

    //progression notice
    if (i%(par.Nsim/100)==0)
      {cout << i/(par.Nsim/100) << "% completed" << endl;}

    //cycle over all particles
    for(k=0;k<tbr.Npart;k++)
    { 
      //save old variables: position x, y, orientation phi and tumbling rate trate
      part.x_old[k]=part.x[k];
      part.y_old[k]=part.y[k];
      part.phi_old[k]=part.phi[k];
      part.trate_old[k]=part.trate[k];
      
      
      //Euler-Maruyama step. 
      //v_red is the reduced activity parameter during collisions
      //par.active_speed is the activity in muM/s
      //part.f* are the forces calculated by interactions
      //par.gamma1 is the friction considering the particles slide on a surface
      //par.Bt = sqrt(2*D*dt)
      dx=(part.v_red[k]*par.active_speed*cos(part.phi[k])+part.fx[k]/par.gamma1)*par.dt+norm_distribution(mt)*par.Bt;
      dy=(part.v_red[k]*par.active_speed*sin(part.phi[k])+part.fy[k]/par.gamma1)*par.dt+norm_distribution(mt)*par.Bt;

      //if (k==0)
      //  cout << "density " << part.density[k] << " v_red " << part.v_red[k]*par.active_speed << " v_real " << sqrt(dx*dx+dy*dy)/par.dt << endl;

      part.x[k]=part.x[k]+dx;
      part.y[k]=part.y[k]+dy;
      
      if (par.ABP_model!=1)
      {
        //Poisson extraction of tumbling.
        rnum=uni_distribution(mt);
        tprob=1.0-exp(-par.dt*part.trate[k]*par.total_tbr_multiplier);

      
        if (rnum<tprob) //When tumbling, chooses the orientation at random
          part.phi[k] = (uni_distribution(mt)-.5)*2*PI;
        else //Otherwise, normal Euler-Maruyama step, Br=sqrt(2*D_r*dt);
          part.phi[k]=part.phi[k]+norm_distribution(mt)*par.Br; 
      }
      else
      {
        part.phi[k]=part.phi[k]+part.torque[k]*par.dt+norm_distribution(mt)*par.Br; //par.c*part.v[k]*(part.vx[k]*sin(part.phi[k])-part.vy[k]*cos(part.phi[k]))
      }

      if (par.S_model>0)
      {
        part.svar[k]+=(par.S_gamma*part.illval[k]*(par.S_stot-part.svar[k])-part.svar[k]/par.S_tau)*par.dt;
        part.smem_state[k]=sgn(par.S_st-part.svar[k]);
      }

    }

    part.clean_forces_trategradients(); //clean arrays of trate gradients and forces

    //Cell list, boundary conditions and forces
    if (par.PBC==0) //No PBC
    {
      
      part.create_cell_list();

      part.slip_boundary_condition(i);//this needs cell lists, in absense of interactions though it is wasteful to create them, so only use INDEPENDENT=1 with PBC=1

      if (par.INDEPENDENT==0)
      {
        part.compute_forces(i);
      }
    }
    else //PBC
    {
      part.periodic_boundary_conditions();

      if (par.INDEPENDENT==0)
      {
        part.create_cell_list();
        part.compute_forces_PBC(i);
      }
    }

    if (par.ABP_model!=1)
    {
      //Tumbling is handled: ie calculation of trate, its gradients and how particles are affected by theta based on the parameters
      if (par.rec_jump>0)
      {
        if (i%par.rec_jump==0)
        {
          part.tumbling_handler(i);
        }
      }
      else
      {
        part.tumbling_handler(i);
      }
    }
    else
    {
      //Torque is handled
      if (par.rec_jump>0)
      {
        if (i%par.rec_jump==0)
        {
          part.torque_handler(i);
        }
      }
      else
      {
        part.torque_handler(i);
      }
    }

    

    //Reduce speed if a collision happens
    if (par.v_reduction==1)
    {
      part.collision_speeds();
    }
    
    part.save_logs(i);
  }

  void MSD()
  {
    int i, j, level, k;
    double dr2, dx, dy, oldvalue;
    level=0;
    for (i=1; i<=par.Nsim/par.print_skip; i*=par.Nblock)
    {
      for(k=0;k<tbr.Npart;k++)
      {
        dx=part.x_log[k+i*tbr.Npart]-part.x_log[k];
        dy=part.y_log[k+i*tbr.Npart]-part.y_log[k];
        dr2=dx*dx+dy*dy;//pow(dx-L*rint(dx/L),2)+pow(dy-L*rint(dy/L),2);
        
        oldvalue=M2.D[level];
        M2.D[level]+=(dr2-oldvalue)/(k+1);
        M2.Dvar[level]+=(dr2-oldvalue)*(dr2-M2.D[level]);
      }
      M2.Dvar[level]=sqrt(M2.Dvar[level])/tbr.Npart;
      level++;
    }
    
  }
  void MSD_timeavg()
  {
    int i, j, k, level;
    double dr2, dx, dy, oldvalue;
    level=0;
    for (i=1; i<=par.Nsim/par.print_skip; i*=par.Nblock)
    {
      for (j=0;j<par.Nsim/par.print_skip/i;j++)
      {
        for (k=0;k<tbr.Npart;k++)
        {
          dx=part.x_log[k+i*(j+1)*tbr.Npart]-part.x_log[k+i*j*tbr.Npart];
          dy=part.y_log[k+i*(j+1)*tbr.Npart]-part.y_log[k+i*j*tbr.Npart];
          dr2=dx*dx+dy*dy;

          oldvalue=M2.D[level];
          M2.D[level]+=(dr2-oldvalue)/(k+j*tbr.Npart+1);
          M2.Dvar[level]+=(dr2-oldvalue)*(dr2-M2.D[level]);
        } 
      }
      M2.Dvar[level]=sqrt(M2.Dvar[level])/(tbr.Npart*par.Nsim/par.print_skip/i);
      level++;
    }
  }
  void MD()
  {
    int i, j, k, level;
    double dx,dy,dr, oldvalue;
    level=0;
    for (i=1; i<=par.Nsim/par.print_skip; i*=par.Nblock)
    {
      for(k=0;k<(tbr.Npart);k++)
      {
        dx=part.x_log[k+i*tbr.Npart]-part.x_log[k];
        dy=part.y_log[k+i*tbr.Npart]-part.y_log[k];
        dr=dx+dy;

        oldvalue=M1.D[level];
        M1.D[level]+=(dr-oldvalue)/(k+1);
        M1.Dvar[level]+=(dr-oldvalue)*(dr-M1.D[level]);
      }
      M1.Dvar[level]=sqrt(M1.Dvar[level])/(tbr.Npart);
      level++;
    }  
  }

  void MD_timeavg()
  {
    int i, j, k, level;
    double dr, dx, dy, oldvalue;
    level=0;
    for (i=1; i<=par.Nsim/par.print_skip; i*=par.Nblock)
    {
      for (j=0;j<par.Nsim/par.print_skip/i;j++)
      {
        for (k=0;k<tbr.Npart;k++)
        {
          dx=part.x_log[k+i*(j+1)*tbr.Npart]-part.x_log[k+i*j*tbr.Npart];
          dy=part.y_log[k+i*(j+1)*tbr.Npart]-part.y_log[k+i*j*tbr.Npart];
          dr=dx+dy;

          oldvalue=M1.D[level];
          M1.D[level]+=(dr-oldvalue)/(k+j*tbr.Npart+1);
          M1.Dvar[level]+=(dr-oldvalue)*(dr-M1.D[level]);
        }
      }
      M1.Dvar[level]=sqrt(M1.Dvar[level])/(tbr.Npart*par.Nsim/par.print_skip/i);
      level++;
    }
  }
  
  void print_log()
  {
    int i, k;
    string filename = "logs//Simlog"+to_string(sim_number)+par.type_name;
    if (par.TBR_I>0)
    {
      filename+="_tbrI";
      if (par.tbr_amplification>1.01 || par.tbr_amplification<.99)
        filename+="_tbramp"+to_string(par.tbr_amplification);
    }
    if (par.TBR_THETA>0)
    {
      filename+="_tbrth";
      if (par.grad_amplification>1.01 || par.grad_amplification<.99)
        filename+="_grdamp"+to_string(par.grad_amplification);
      if (par.grad_decay_rate>0.0)
        filename+="_grddec"+to_string(par.grad_decay_rate);
      if (par.grad_sign_thresh>0)
      {
        filename+="_grsgn"+to_string(par.grad_sign_thresh);
        if (par.threshold_std>0)
          filename+="_thrstd"+to_string(par.threshold_std);
        if (par.threshold_decay_rate!=0)
          filename+="_thrdec"+to_string(par.threshold_decay_rate);
      }
      if (par.grad_sign_dist>0)
        filename+="_grdstd"+to_string(par.grad_sign_dist*par.part_diam);
      if (par.grad_dependence>0)
        filename+="_grdep"+to_string(par.grad_dependence);
    }
    if (par.TBR_DIST>0)
      filename+="_tbrdist";
    if (par.COLLISIONS>0)
      filename+="_coll"+to_string(par.coll_tbr_rate);
    if (par.INDEPENDENT>0)
      filename+="_ind";
    if (par.v_reduction<1)
      filename+="_vred"+to_string(par.v_reduction);
    if (par.upper_mem_thresh>0)
      filename+="_mem"+to_string(par.upper_mem_thresh);
    if (par.time_mem>0)
      filename+="_tmem"+to_string(par.time_mem);
    if (par.DELAY==1)
      filename+="_del";
    if (par.fk>0)
      filename+="_fk"+to_string(par.fk);
    if (par.rec_jump>0)
      filename+="_rcjp"+to_string(par.rec_jump);
    if (par.NOCENTER>0)
      filename+="_noc"+to_string(par.NOCENTER);
    if (par.S_model>0)
      filename+="_smod_tau"+to_string(par.S_tau)+"_gam"+to_string(par.S_gamma)+"_st"+to_string(par.S_st)+"_stot"+to_string(par.S_stot);
    if (par.ABP_model==1)
      filename+="_abp_torque"+to_string(par.ABP_torque_fac);
    if (par.sub_dist>.51 || par.sub_dist<.49)
      filename+="_sub_dist"+to_string(par.sub_dist);
    if (par.Nlevel!=18)
      filename+="_Nl"+to_string(par.Nlevel);
    if (par.active_speed*par.part_diam>50.1 || par.active_speed*par.part_diam<49.9 )
      filename+="_v0"+to_string(par.active_speed*par.part_diam);
    if (par.total_tbr_multiplier<.99 || par.total_tbr_multiplier>1.01)
      filename+="_tbm"+to_string(par.total_tbr_multiplier);
    if (par.dt<.0009 || par.dt>0.0011)
      filename+="_dt"+to_string(par.dt);
    if (par.eta/par.part_diam<2.09 || par.eta/par.part_diam>2.11)
      filename+="_eta"+to_string(par.eta/par.part_diam);
    filename+="_rho"+to_string(par.rho);
    filename+=".dat";
    par.print_parameters(filename);
    tbr.print_tbr_parameters(filename);
    ofstream file_out(filename, std::ofstream::out | std::ofstream::app);
  
    file_out.precision(16);

    for (i=0; i<=par.Nsim/par.print_skip; i++)
    {
      for (k=0;k<tbr.Npart;k++)
      {
        file_out << par.dt*i*par.print_skip << " " << k << " " << part.x_log[k+i*tbr.Npart] << " " << part.y_log[k+i*tbr.Npart] << " " << part.phi_log[k+i*tbr.Npart] << " " << part.trate_log[k+i*tbr.Npart] << " " << part.state_log[k+i*tbr.Npart] << endl;
      }    
    }
  
    file_out.close();
  }

  void print_quantities()
  {
    int i, k;

    string filename = "logs//Simqts"+to_string(sim_number)+par.type_name;
    if (par.TBR_I>0)
    {
      filename+="_tbrI";
      if (par.tbr_amplification>1.01 || par.tbr_amplification<.99)
        filename+="_tbramp"+to_string(par.tbr_amplification);
    }
    if (par.TBR_THETA>0)
    {
      filename+="_tbrth";
      if (par.grad_amplification>1.01 || par.grad_amplification<.99)
        filename+="_grdamp"+to_string(par.grad_amplification);
      if (par.grad_decay_rate>0.0)
        filename+="_grddec"+to_string(par.grad_decay_rate);
      if (par.grad_sign_thresh>0)
      {
        filename+="_grsgn"+to_string(par.grad_sign_thresh);
        if (par.threshold_std>0)
          filename+="_thrstd"+to_string(par.threshold_std);
        if (par.threshold_decay_rate!=0)
          filename+="_thrdec"+to_string(par.threshold_decay_rate);
      }
      if (par.grad_sign_dist>0)
        filename+="_grdstd"+to_string(par.grad_sign_dist*par.part_diam);
      if (par.grad_dependence>0)
        filename+="_grdep"+to_string(par.grad_dependence);
    }
    if (par.TBR_DIST>0)
      filename+="_tbrdist";
    if (par.COLLISIONS>0)
      filename+="_coll"+to_string(par.coll_tbr_rate);
    if (par.INDEPENDENT>0)
      filename+="_ind";
    if (par.v_reduction<1)
      filename+="_vred"+to_string(par.v_reduction);
    if (par.upper_mem_thresh>0)
      filename+="_mem"+to_string(par.upper_mem_thresh);
    if (par.time_mem>0)
      filename+="_tmem"+to_string(par.time_mem);
    if (par.DELAY==1)
      filename+="_del";
    if (par.fk>0)
      filename+="_fk"+to_string(par.fk);
    if (par.rec_jump>0)
      filename+="_rcjp"+to_string(par.rec_jump);
    if (par.NOCENTER>0)
      filename+="_noc"+to_string(par.NOCENTER);
    if (par.S_model>0)
      filename+="_smod_tau"+to_string(par.S_tau)+"_gam"+to_string(par.S_gamma)+"_st"+to_string(par.S_st)+"_stot"+to_string(par.S_stot);
    if (par.ABP_model==1)
      filename+="_abp_torque"+to_string(par.ABP_torque_fac);
    if (par.sub_dist>.51 || par.sub_dist<.49)
      filename+="_sub_dist"+to_string(par.sub_dist);
    if (par.Nlevel!=18)
      filename+="_Nl"+to_string(par.Nlevel);
    if (par.active_speed*par.part_diam>50.1 || par.active_speed*par.part_diam<49.9 )
      filename+="_v0"+to_string(par.active_speed*par.part_diam);
    if (par.total_tbr_multiplier<.99 || par.total_tbr_multiplier>1.01)
      filename+="_tbm"+to_string(par.total_tbr_multiplier);
    if (par.dt<.0009 || par.dt>0.0011)
      filename+="_dt"+to_string(par.dt);
    if (par.eta/par.part_diam<2.09 || par.eta/par.part_diam>2.11)
      filename+="_eta"+to_string(par.eta/par.part_diam);
    filename+="_rho"+to_string(par.rho);
    filename+=".dat";
    par.print_parameters(filename);
    tbr.print_tbr_parameters(filename);
    ofstream file_out(filename, std::ofstream::out | std::ofstream::app);
  
    file_out.precision(16);

    for (i=0; i<=par.Nsim/par.print_skip; i++)
    {
      file_out << par.dt*i*par.print_skip << " " << part.pot[i*par.print_skip] << " " << part.vir[i*par.print_skip] << endl;   
    }
  
    file_out.close();
  }
};

struct averaging_data{
  valarray<double> arraytot;
  valarray<double> arrayvartot;
  valarray<double> oldarray;

  string name;
  parameters par;

  void incremental_average(simulation sim, moment M)
  {
    oldarray=arraytot;
    arraytot+=(M.D-oldarray)/(sim.sim_number+1.0);
    arrayvartot+=(M.Dvar-arrayvartot)/(sim.sim_number+1.0);///(M.D-oldarray)*(M.D-arraytot);
  }

  void normalize_variance(double N)
  {
    arrayvartot/=sqrt((double)N);//sqrt(arrayvartot/((double)N))/sqrt((double)N);
  }

  void print_array()
  {
    int i, level;
    string filename;
    filename = name+".dat";
  
    par.print_parameters(filename);
    
    ofstream file_out(filename, std::ofstream::out | std::ofstream::app);
  
    file_out.precision(16);
    level=0;

    for (i=1; i<=par.Nsim/par.print_skip; i*=par.Nblock)
    {
      file_out << par.dt*i*par.print_skip << " " << arraytot[level] << " " << arrayvartot[level] << endl;
      level++;
    }

    file_out.close();
  }


};



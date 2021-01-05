#include "base/kaldi-common.h"
#include "cudamatrix/cu-allocator.h"
#include "nnet3/nnet-chain-training.h"
#include "util/common-utils.h"

#include "chain/chain-den-graph.h"
#include "chain/chain-training.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-training.h"
#include "nnet3/nnet-utils.h"
#include "chain/chain-numerator.h"
#include "chain/chain-denominator.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace kaldi::chain;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+chain neural network parameters with backprop and "
        "stochastic\n"
        "gradient descent.  Minibatches are to be created by "
        "nnet3-chain-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best "
        "to\n"
        "use it with a GPU).\n"
        "\n"
        "Usage:  nnet3-chain-train [options] <raw-nnet-in> "
        "<denominator-fst-in> <chain-training-examples-in> <raw-nnet-out>\n"
        "\n"
        "nnet3-chain-train 1.raw den.fst 'ark:nnet3-merge-egs 1.cegs ark:-|' "
        "2.raw\n";

    int32 srand_seed = 0;
    int32 total_train_steps = 1;
    bool binary_write = false;
    std::string use_gpu = "yes";
    NnetChainTrainingOptions opts;

    ParseOptions po(usage);
    po.Register("total_train_steps", &total_train_steps, "total_train_steps for this single eg ");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    opts.Register(&po);
    RegisterCuAllocatorOptions(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
                den_fst_rxfilename = po.GetArg(2),
                examples_rspecifier = po.GetArg(3),
                nnet_wxfilename = po.GetArg(4),
                delta_nnet_wxfilename = po.GetArg(5);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    bool ok;

    {
      fst::StdVectorFst den_fst;
      ReadFstKaldi(den_fst_rxfilename, &den_fst);

      SequentialNnetChainExampleReader example_reader(examples_rspecifier);

      chain::DenominatorGraph den_graph_(den_fst, nnet.OutputDim("output"));
      const NnetChainTrainingOptions opts_(opts);
      CachingOptimizingCompiler compiler_(nnet,
                                          opts_.nnet_config.optimize_config,
                                          opts_.nnet_config.compiler_config);
      int32 num_minibatches_processed_ = 0;
      int32 srand_seed_ = RandInt(0, 100000);
      if (opts.nnet_config.zero_component_stats)
        ZeroComponentStats(&nnet);
      KALDI_ASSERT(opts.nnet_config.momentum >= 0.0 &&
                   opts.nnet_config.max_param_change >= 0.0 &&
                   opts.nnet_config.backstitch_training_interval > 0);
      Nnet *delta_nnet_ = nnet.Copy();
      ScaleNnet(0.0, delta_nnet_);
      const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
      // stats for max-change.
      std::vector<int32> num_max_change_per_component_applied_;
      int32 num_max_change_global_applied_;
      num_max_change_per_component_applied_.resize(num_updatable, 0);
      num_max_change_global_applied_ = 0;
      int i = 0; 
      const NnetChainExample eg = example_reader.Value();
      // for (; !example_reader.Done(); example_reader.Next())
      for (int step = 0; step < total_train_steps; step++)
      {
        std::cout<< "train step : " << step << std::endl;
        // function NnetChainTrainer::Train
        // const NnetChainExample eg = example_reader.Value();
        bool need_model_derivative = true;
        const NnetTrainerOptions &nnet_config = opts_.nnet_config;
        bool use_xent_regularization = (opts_.chain_config.xent_regularize != 0.0);
        std::cout << "use_xent_regularization: " << use_xent_regularization << std::endl;
        ComputationRequest request;
        // GetChainComputationRequest(*nnet_, chain_eg, need_model_derivative,
        GetChainComputationRequest(nnet, eg, need_model_derivative,
                                   nnet_config.store_component_stats,
                                   use_xent_regularization, need_model_derivative,
                                   &request);
        // const NnetComputation *computation = compiler_.Compile(request).get();
				std::shared_ptr<const NnetComputation> computation =
						compiler_.Compile(request);
      
        // function NnetChainTrainer::TrainInternal
        {
          NnetComputer computer(nnet_config.compute_config, *computation,
                                &nnet, delta_nnet_);
          computer.AcceptInputs(nnet, eg.inputs);
          computer.Run();
          // function NnetChainTrainer::ProcessOutputs
					{
            bool is_backstitch_step2 = false;
            const std::string suffix = (is_backstitch_step2 ? "_backstitch" : "");
            std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
                end = eg.outputs.end();
            for (; iter != end; ++iter) {
              const NnetChainSupervision &sup = *iter;
              int32 node_index = nnet.GetNodeIndex(sup.name);
              if (node_index < 0 ||
                  !nnet.IsOutputNode(node_index))
                  KALDI_ERR << "Network has no output named " << sup.name;
     
              const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput(sup.name);
              // std::cout << nnet_output << std::endl;
              CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                                    nnet_output.NumCols(),
                                                    kUndefined);
              // std::cout << "nnet_output: " << nnet_output << std::endl; 
              std::cout << "nnet_output.NumRows(): " << nnet_output.NumRows() << " NumCols: " << nnet_output.NumCols() << std::endl; 
              Matrix<kaldi::BaseFloat> cpu_nnet_output(nnet_output.NumRows(), nnet_output.NumCols());
              nnet_output.CopyToMat(&cpu_nnet_output);
              std::string nnet_output_wspecifier = "ark,t:nnet_output.ark";
              BaseFloatMatrixWriter nnet_output_writer(nnet_output_wspecifier);
              nnet_output_writer.Write("lala", cpu_nnet_output);

              BaseFloat tot_objf, tot_l2_term, tot_weight;
              // function ComputeChainObjfAndDeriv
              /*
              const ChainTrainingOptions &opts = opts_.chain_config;
              BaseFloat num_logprob_weighted, den_logprob_weighted;
              bool ok = true;
              nnet_output_deriv.SetZero();
              { // Doing the denominator first helps to reduce the maximum
                // memory use, as we can set 'xent_deriv' to nonempty after
                // we've freed the memory in this object.
                DenominatorComputation denominator(opts, den_graph_,
                                                   sup.supervision.num_sequences,
                                                   nnet_output);
            
                den_logprob_weighted = sup.supervision.weight * denominator.Forward();
                ok = denominator.Backward(-sup.supervision.weight,
                                          &nnet_output_deriv);
              }
              Matrix<kaldi::BaseFloat> deno_cpu_deriv(nnet_output_deriv.NumRows(), nnet_output_deriv.NumCols());
              nnet_output_deriv.CopyToMat(&deno_cpu_deriv);
              // std::cout << "nnet_output_deriv: " << cpu_deriv << std::endl;
              std::string deno_wspecifier = "ark,t:deno_cal_grad_cpu_deriv.ark";
              BaseFloatMatrixWriter deno_cpu_deriv_writer(deno_wspecifier);
              deno_cpu_deriv_writer.Write("lala", deno_cpu_deriv);

              {
                NumeratorComputation numerator(sup.supervision, nnet_output);
                // note: supervision.weight is included as a factor in the derivative from
                // the numerator object, as well as the returned logprob.
                num_logprob_weighted = numerator.Forward();
            
                numerator.Backward(&nnet_output_deriv);
              }
              tot_objf = num_logprob_weighted - den_logprob_weighted;
              tot_weight = sup.supervision.weight * sup.supervision.num_sequences *
                  sup.supervision.frames_per_sequence;

              std::cout << "den_logprob_weighted: " << den_logprob_weighted << std::endl;

              std::cout << "num_logprob_weighted: " << num_logprob_weighted << std::endl;
              std::cout << "tot_objf: " << tot_objf << std::endl;
              std::cout << "tot_weight: " << tot_weight << std::endl;
              */



              ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                                       sup.supervision, nnet_output,
                                       &tot_objf, &tot_l2_term, &tot_weight,
                                       &nnet_output_deriv,
                                       NULL);
	      std::cout << "tot_objf: " << tot_objf << std::endl;
	      std::cout << "tot_l2_term: " << tot_l2_term << std::endl;
	      std::cout << "tot_weight: " << tot_weight << std::endl;
              Matrix<kaldi::BaseFloat> cpu_deriv(nnet_output_deriv.NumRows(), nnet_output_deriv.NumCols());
              nnet_output_deriv.CopyToMat(&cpu_deriv);
              // std::cout << "nnet_output_deriv: " << cpu_deriv << std::endl;
              std::string wspecifier = "ark,t:cal_grad_cpu_deriv.ark";
              BaseFloatMatrixWriter cpu_deriv_writer(wspecifier);
              cpu_deriv_writer.Write("lala", cpu_deriv);
              computer.AcceptInput(sup.name, &nnet_output_deriv);
            } // end of a single eg
					} // end of function NnetChainTrainer::ProcessOutputs
          computer.Run();

          WriteKaldiObject(*delta_nnet_, delta_nnet_wxfilename, binary_write);

          bool success = UpdateNnetWithMaxChange(*delta_nnet_,
              nnet_config.max_param_change, 1.0, 1.0 - nnet_config.momentum, &nnet,
              &num_max_change_per_component_applied_, &num_max_change_global_applied_);
          std::cout << "update success" << success << std::endl;
          // if (success)
          //   ScaleNnet(nnet_config.momentum, delta_nnet_);
          // else
          //   ScaleNnet(0.0, delta_nnet_);
        } // end of function NnetChainTrainer::TrainInternal
        // if (num_minibatches_processed_ == 0) {
        //   ConsolidateMemory(&nnet);
        //   ConsolidateMemory(delta_nnet_);
        // }
        num_minibatches_processed_++;
      } // end of all egs
      // std::cout << "hello world  lala lala srand_seed " << srand_seed_ << std::endl;
    }

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Wrote raw model to " << nnet_wxfilename;
    return (ok ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

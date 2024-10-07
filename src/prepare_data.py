import os
import sys
from nvidia.dali.plugin.jax import DALIGenericIterator
from data import directory_sequence_pipe, DirectorySequence

def main(argv) -> None:

    source_grid = os.path.abspath(argv[0])
    source_tipsy = os.path.abspath(argv[1])

    destination_grid = os.path.abspath(argv[2])

    input_grid_size = int(argv[3])
    output_grid_size = int(argv[4])

    dataset_params = {
        "grid_size" : input_grid_size,
        "grid_directory" : source_grid,
        "tipsy_directory" : source_tipsy,
        "start" : 0,
        "steps" : 5,
        "stride" : 1,
        "flip" : False,
        "type" : "all",
        "normalize" : False}
    
    train_dataset = DirectorySequence(**dataset_params)
    train_data_pipeline = directory_sequence_pipe(train_dataset, output_grid_size)
    train_data_iterator = DALIGenericIterator(train_data_pipeline, ["data", "step", "attributes"])

    index = 0
    # iterate over the data
    for i, batch in enumerate(train_data_iterator):
        # shape of batch [Batch, Frames, Channels, Depth, Height, Width]
        # channels are rho, vx, vy, vz

        datafields = batch["data"]
        time = batch["step"]

        batch_size = datafields.shape[0]

        for j in range(batch_size):
            rho = datafields[j, :, 0, :, :, :]
            vx = datafields[j, :, 1, :, :, :]
            vy = datafields[j, :, 2, :, :, :]
            vz = datafields[j, :, 3, :, :, :]
            time = time[j]

            sequence_length = rho.shape[0]

            # folder name is the index with three digits
            folder_name = f"{index:03d}"

            output_folder = os.path.join(destination_grid, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            time_file = os.path.join(output_folder, f"time_{k:03d}.bin")
            time[k].tofile(time_file)

            print(f"Writing sequence {index} to {output_folder}")

            for k in range(sequence_length):
                rho_file = os.path.join(output_folder, f"rho_{k:03d}.bin")
                vx_file = os.path.join(output_folder, f"vx_{k:03d}.bin")
                vy_file = os.path.join(output_folder, f"vy_{k:03d}.bin")
                vz_file = os.path.join(output_folder, f"vz_{k:03d}.bin")
                # write as binary files
                rho[k].tofile(rho_file)
                vx[k].tofile(vx_file)
                vy[k].tofile(vy_file)
                vz[k].tofile(vz_file)
            
            index += 1


                

    
if __name__ == "__main__":
    main(sys.argv[1:])
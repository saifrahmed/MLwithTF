package cifar10.classifier.saif.cifar10.adapters.recyclerview;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.bumptech.glide.request.target.BitmapImageViewTarget;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.List;

import cifar10.classifier.saif.cifar10.R;
import cifar10.classifier.saif.cifar10.retrofit.ImagesService;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * Created by shams on 11/12/2016.
 */

public class FoundImagesRecyclerViewAdapter extends RecyclerView.Adapter<FoundImagesRecyclerViewAdapter.ViewHolder> {

    LayoutInflater mLayoutInflater;
    Context mContext;
    List<String> mImagesUrls;
    ImagesService mImagesService;
    TextView mClassifiedImageTV;

    public FoundImagesRecyclerViewAdapter(Context context, List<String> imagesUrls, ImagesService imagesService,
                                          TextView classifiedImageTV) {
        this.mContext = context;
        this.mImagesUrls = imagesUrls;
        this.mLayoutInflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        this.mImagesService = imagesService;
        this.mClassifiedImageTV = classifiedImageTV;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        return new ViewHolder(mLayoutInflater.inflate(R.layout.images_list_recyclerview_item, parent, false));
    }

    @Override
    public void onBindViewHolder(final ViewHolder holder, int position) {
        Glide.with(mContext)
                .load(mImagesUrls.get(position))
                .asBitmap()
                .into(new BitmapImageViewTarget(holder.mItemImageView) {
                    @Override
                    protected void setResource(Bitmap resource) {
                        // Do bitmap magic here
                        super.setResource(resource);
                    }
                });

        holder.mItemImageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap = ((BitmapDrawable) holder.mItemImageView.getDrawable()).getBitmap();

                Bitmap resized = Bitmap.createScaledBitmap(bitmap, 32, 32, true);

                // Assume block needs to be inside a Try/Catch block.
                String path = Environment.getExternalStorageDirectory().toString();
                OutputStream fOut;
                File file = new File(path, "temp_image.jpg"); // the File to save , append increasing numeric counter to prevent files from getting overwritten.
                try {
                    fOut = new FileOutputStream(file);

                    resized.compress(Bitmap.CompressFormat.JPEG, 85, fOut); // saving the Bitmap to a file compressed as a JPEG with 85% compression rate
                    fOut.flush(); // Not really required
                    fOut.close(); // do not forget to close the stream

                    MediaStore.Images.Media.insertImage(mContext.getContentResolver(), file.getAbsolutePath(), file.getName(), file.getName());

                    MultipartBody.Part filePart =
                            MultipartBody.Part.createFormData("file", file.getName(),
                                    RequestBody.create(MediaType.parse("image/*"), file));

                    mImagesService.classifyImage(filePart).enqueue(new Callback<String>() {
                        @Override
                        public void onResponse(Call<String> call, Response<String> response) {
                            mClassifiedImageTV.setText(response.body());
                        }

                        @Override
                        public void onFailure(Call<String> call, Throwable t) {
                            Toast.makeText(mContext, "Couldn't classify image!", Toast.LENGTH_LONG).show();
                        }
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    @Override
    public int getItemCount() {
        return mImagesUrls == null ? 0 : mImagesUrls.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {

        ImageView mItemImageView;

        public ViewHolder(View itemView) {
            super(itemView);

            mItemImageView = (ImageView) itemView.findViewById(R.id.image_item_imageview);
        }
    }
}

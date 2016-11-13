package cifar10.classifier.saif.cifar10;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import cifar10.classifier.saif.cifar10.adapters.recyclerview.FoundImagesRecyclerViewAdapter;
import cifar10.classifier.saif.cifar10.retrofit.ImagesListResponse;
import cifar10.classifier.saif.cifar10.retrofit.ImagesService;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MainActivity extends AppCompatActivity {

    ImagesService mImagesService;
    EditText mSearchStringET;
    RecyclerView mImagesListRV;
    TextView mClassifiedImageTV;
    Button mSearchImagesB;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mSearchStringET = (EditText) findViewById(R.id.input_search);
        mImagesListRV = (RecyclerView) findViewById(R.id.images_list_recyclerview);
        mClassifiedImageTV = (TextView) findViewById(R.id.classified_images_textview);
        mSearchImagesB = (Button) findViewById(R.id.search_images_button);

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://ec2-52-23-154-153.compute-1.amazonaws.com:5000")
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        mImagesService = retrofit.create(ImagesService.class);

        mImagesListRV.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        mImagesListRV.setAdapter(new FoundImagesRecyclerViewAdapter(this, null, mImagesService, mClassifiedImageTV));

        mSearchImagesB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final String searchText = mSearchStringET.getText().toString();

                if (searchText.length() > 0) {
                    mImagesService.findImages(searchText.trim()).enqueue(new Callback<ImagesListResponse>() {
                        @Override
                        public void onResponse(Call<ImagesListResponse> call, Response<ImagesListResponse> response) {
                            mImagesListRV.setAdapter(new FoundImagesRecyclerViewAdapter(MainActivity.this,
                                    response.body().images_list, mImagesService, mClassifiedImageTV));
                        }

                        @Override
                        public void onFailure(Call<ImagesListResponse> call, Throwable t) {
                            Toast.makeText(MainActivity.this, "Couldn't find images for \"" + searchText + "\"!",
                                    Toast.LENGTH_LONG).show();
                        }
                    });
                }
            }
        });

        if (Build.VERSION.SDK_INT >= 23) {
            if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                Log.v("Cifar-10","Permission is granted");
            } else {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 100);
            }
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(grantResults[0]== PackageManager.PERMISSION_GRANTED){
            Log.v("Cifar-10","Permission: "+permissions[0]+ "was "+grantResults[0]);
            //resume tasks needing this permission
        } else {
            finish();
        }
    }
}

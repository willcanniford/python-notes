# Basic Flask Notes 

Official documentation: [Flask Quickstart](https://flask.palletsprojects.com/en/1.1.x/quickstart/)

Note that you have to, somewhere on your system mark that the directory is a `FLASK_APP`.  
You can do this on mac using the `export` command. 

- - - -

## Debug mode
If you turn on the debug mode for the application, then you don't have to restart the server everytime that you want the changes that you've made to the files to take effect. If you don't do this then refreshing the page won't change the `localhost` server instance to update the visual display of the application.  

`export FLASK_ENV=development` can be run to set the current environment to debug mode.   
`export FLASK_ENV=production` can be run to turn this off again. 


- - - -

## TailwindCSS

`npx tailwind build static/css/build.css -o static/css/main.css` this uses npm to build the static css file for tailwind from the a given css file `static/css/build.css` and outputs it to the `static/css/main.css` which is then referenced in the code. 